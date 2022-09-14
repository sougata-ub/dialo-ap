""" Parse the out-of-domain CMV posts using both recreated Ampersand pipeline (AmpersandArgParser)
    and our pipeline (ArgParser) """
import os
import torch
import pickle
import pandas as pd
import numpy as np
from parser import ArgParser
from ampersand_parsing_pipeline import AmpersandArgParser
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    configurations = {'in_dim': 768, 'out_dim': 600, 'n_local_classes': 3, 'n_global_classes': 3, 'n_context_classes': 3,
                      'n_token_classes': 2, 'n_type_classes': 4, 'n_layers': 4, 'n_heads': 4, 'n_self_attn_layers': 1,
                      'n_ctx_attn_layers': 2, 'self_mha': True, 'predict_contextual_relationship': True,
                      'context_attention': True, 'predict_segments': True, 'predict_edu_type': True,
                      'predict_local_relations': True, 'predict_global_relations': True,
                      'base_transformer': 'roberta-base', 'lr': 1e-05, 'batch_size': 1, 'accumulation': 4,
                      'device_num': 0, 'sigmoid_threshold': 0.5,
                      'training_file': '<file_name.pkl>',
                      'num_epochs': 15, 'increase_token_type': False, 'debug_mode': False, 'run_curriculum': 'C_target_dataset',
                      'n_rerun': 1, 'increase_positional_embeddings': 'true', 'num_workers': 4, 'use_gpu': True,
                      'pre_training_file': '<file_name.pkl>',
                      'pre_train': False, 'pretrained_model': None, 'use_pretrained_base': False,
                      'curriculum_model': '<file_name.pt>',
                      'use_curriculum_base': 'true', 'positive_class_weight': 3.0, 'global_positive_class_weight': 3.0, 'n_examples': None,
                      'experiment_desc': 'local_global_equal_loss_preference_local_global_equal_loss_weight_1650858092',
                      'experiment_number': '13', 'loss_local_rel_head_wt': 0.4, 'loss_local_rel_deprel_wt': 0.022499999999999992,
                      'loss_edu_type_wt': 0.1, 'loss_global_rel_head_wt': 0.4, 'loss_global_rel_deprel_wt': 0.022499999999999992,
                      'loss_ctx_rel_deprel_wt': 0.01, 'loss_local_rel_head_app_wt': 0.022499999999999992,
                      'loss_segmentation_wt': 0.022499999999999992, 'early_stopping': 2, 'skip_training': False,
                      'run_inference': False, 'train_distributed': False,
                      'model_name': '<file_name.pt>',
                      'log_name': '<file_name.json>',
                      'accumulate': 8}
    model_file = "<file_name.pt>"
    parsing_model = ArgParser(configurations, model_file, evaluate=False, debug=False)
    path = "/home/argumentation/reddit-argument-parser/results/model_backups/"

    component_pretrained = path + "bert-base-uncased_imho_component_model.pt"
    inter_pretrained = path + "bert-base-uncased_iqr_inter_model.pt"
    intra_pretrained = path + "bert-base-uncased_imho_intra_model.pt"
    device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else "cpu"
    amp_parsing_model = AmpersandArgParser(device, component_pretrained, intra_pretrained, inter_pretrained, debug=False)
    cmv_data = pickle.load(open("./data/cmv_scraped_conversations_filtered.pkl", "rb"))
    print("CMV Data Len:", len(cmv_data))

    from tqdm import tqdm

    annotation_dict = {}

    for k, v in tqdm(cmv_data.items()):
        try:
            parsing_model.configuration["sigmoid_threshold_global"] = 0.5
            parsed_op_05_thresh = parsing_model.run_v2(v["turns"])

            parsing_model.configuration["sigmoid_threshold_global"] = 0.1
            parsed_op_01_thresh = parsing_model.run_v2(v["turns"])

            parsing_model.configuration["sigmoid_threshold_global"] = 0.2
            parsed_op_02_thresh = parsing_model.run_v2(v["turns"])

            parsing_model.configuration["sigmoid_threshold_global"] = 0.05
            parsed_op_005_thresh = parsing_model.run_v2(v["turns"])

            parsing_model.configuration["sigmoid_threshold_global"] = 0.01
            parsed_op_001_thresh = parsing_model.run_v2(v["turns"])

            amp_parsed_op = amp_parsing_model.run_v2(v["turns"])
            annotation_dict[k] = {"parsed_op_05_thresh": parsed_op_05_thresh,
                                  "parsed_op_01_thresh": parsed_op_01_thresh,
                                  "parsed_op_02_thresh": parsed_op_02_thresh,
                                  "parsed_op_005_thresh": parsed_op_005_thresh,
                                  "parsed_op_001_thresh": parsed_op_001_thresh,
                                  "amp_parsed_op": amp_parsed_op}
        except Exception as e:
            pass
    print(len(annotation_dict), "annotations done!")
    pickle.dump(annotation_dict, open("./data/cmv_scraped_conversations_with_inference_filtered.pkl", "wb"))


if __name__ == '__main__':
    main()
