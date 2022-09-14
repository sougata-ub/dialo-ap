""" Use our pre-trained model pipeline to generate automatic C/P/NA tags for the QR & Args.me pre-training dataset """
import json
import pickle
from tqdm import tqdm
import pandas as pd
from parser import ArgParser
import numpy as np
import torch
import torch.nn.functional as F
import copy
import requests
import re
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, SequentialSampler
from torch.cuda.amp import autocast
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


def main():
    device_num = 0

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device_num", type=int, help="CUDA device number", default=device_num)
    parser.add_argument("--data_split", type=str, help="data_split")
    argv = parser.parse_args()
    device_num = int(argv.device_num)
    data_split = argv.data_split

    configurations = {'in_dim': 768, 'out_dim': 600,'n_local_classes': 4,'n_global_classes': 3,'n_context_classes': 3,
                      'n_token_classes': 2, 'n_type_classes': 4, 'n_layers': 4, 'n_heads': 4, 'n_self_attn_layers': 1,
                      'n_ctx_attn_layers': 2, 'self_mha': True, 'predict_contextual_relationship': True,
                      'context_attention': True, 'predict_segments': True, 'predict_edu_type': True,
                      'predict_local_relations': True, 'predict_global_relations': True,
                      'base_transformer': 'roberta-base', 'lr': 1e-05, 'batch_size': 6, 'accumulation': 4,
                      'device_num': device_num, 'sigmoid_threshold': 0.5, 'interpolation': 0.8,
                      'training_file':  '<file_name.pkl>',
                      'num_epochs': 15, 'increase_token_type': False, 'debug_mode': False, 'train_distributed': False,
                      'run_curriculum': 'C_target_dataset', 'n_rerun': 1, 'increase_positional_embeddings': True, 'num_workers': 4,
                      'model_name': '<file_name.pt>',
                      'use_gpu': True, 'pre_training_file': '<file_name.pkl>',
                      'pre_train': False, 'pretrained_model': None, 'use_pretrained_base': False, 'curriculum_model': '<file_name.pt>',
                      'use_curriculum_base': True, 'positive_class_weight': 5.0}

    parsing_model = ArgParser(configurations, '<file_name.pt>', evaluate=False)

    parsing_model.configuration["predict_global_relations"] = False
    parsing_model.configuration["predict_local_relations"] = False
    parsing_model.configuration["predict_contextual_relationship"] = False
    parsing_model.configuration["predict_segments"] = False

    data_lst_dict = pickle.load(open("./ann_data_lst_dict.pkl", "rb"))
    data_lst = data_lst_dict[data_split]

    to_remove = list(set([i[0].split(":")[0] for i in data_lst if len(i[3]) == 0]))
    if len(to_remove) > 0:
        data_lst = [i for i in data_lst if i[0].split(":")[0] not in to_remove]

    dataloader = DataLoader(data_lst, batch_size=64, sampler=SequentialSampler(data_lst),
                            collate_fn=parsing_model.generate_batch)

    annotations = {}
    print("\n::::::: RUNNING ANNOTATIONS FOR SPLIT:", data_split, ":::::::\n")
    for ix, batch in tqdm(enumerate(dataloader)):
        ids, input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask, ctx_type_id, \
            segment_labels, component_labels, local_rel, local_rel_app, local_rel_labels, global_rel, \
                global_rel_labels, global_ctx_adu_labels = batch

        with autocast():
            with torch.no_grad():
                output_dct = parsing_model.parser.predict(input_ids.to(parsing_model.device),
                                                          position_ids=pos_ids.to(parsing_model.device),
                                                          attention_mask=attention_mask.to(parsing_model.device),
                                                          edu_idx=pos_arr.to(parsing_model.device))

        x = zip(torch.argmax(output_dct["logits_edu_type"], -1).tolist(), pos_arr.T.tolist())
        for ix2, i in enumerate(x):
            inpt_ids = [j for j in input_ids[ix2].view(-1).tolist() if j != parsing_model.tokenizer.pad_token_id]
            edu_lbls = [parsing_model.type_tags_rev[j[0]] for j in zip(i[0], i[1]) if j[1] != -1]
            annotations[ids[ix2]] = {"input_ids": inpt_ids, "edu_tags": edu_lbls}

    fname = "./"+data_split+"_annotations.pkl"
    print("\n======== ANNOTATIONS FOR SPLIT", data_split, "DONE! FILE SAVED TO LOCATION:", fname, "========\n")
    pickle.dump(annotations, open(fname, "wb"))


if __name__ == '__main__':
    main()
