""" Code for pretraining BERT with MLM & NSP tasks on the IMHO & QP datasets. """
import math
import pickle
import pandas as pd
from tqdm import tqdm
import re
from transformers import BertTokenizerFast, RobertaTokenizerFast
from transformers import BertForPreTraining
import transformers
import argparse
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import os
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


class DataCollator:
    def __init__(self, tokenizer, device, batch_size=64, num_workers=4, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def pad_list_2d(self, batch, pad_token=None):
        pad_token = self.tokenizer.pad_token_id if pad_token is None else pad_token
        maxlen = max([len(i) for i in batch])
        padded_lst = []

        for i in batch:
            padded_lst.append(i + [pad_token] * (maxlen - len(i)))
        return torch.tensor(padded_lst)

    def generate_batch(self, data_batch):
        inpt_ids, typ_ids, attn_masks, nsp_lbls, mlm_lbls = [], [], [], [], None
        for (input_ids, token_type_ids, attention_mask, nsp_labels) in data_batch:
            inpt_ids.append(input_ids)
            typ_ids.append(token_type_ids)
            attn_masks.append(attention_mask)
            nsp_lbls.append(nsp_labels)

        inpt_ids = self.pad_list_2d(inpt_ids)
        typ_ids = self.pad_list_2d(typ_ids)
        attn_masks = self.pad_list_2d(attn_masks)
        inpt_ids, mlm_lbls = self.mask_tokens(inpt_ids)
        nsp_lbls = torch.tensor(nsp_lbls)
        # return inpt_ids.to(self.device), typ_ids.to(self.device), attn_masks.to(self.device), \
        #        nsp_lbls.to(self.device), mlm_lbls.to(self.device)
        return inpt_ids, typ_ids, attn_masks, nsp_lbls, mlm_lbls

    def data_process(self, data_dict):
        data_lst = []
        for ix, i in enumerate(data_dict["input_ids"]):
            data_lst.append([i, data_dict["token_type_ids"][ix], data_dict["attention_mask"][ix],
                             data_dict["nsp_labels"][ix]])
        return data_lst

    def get_dataloader(self, data_dict, training=True):
        dataset = self.data_process(data_dict)
        if training:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=DistributedSampler(dataset),
                                    num_workers=self.num_workers, collate_fn=self.generate_batch)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset),
                                    num_workers=self.num_workers, collate_fn=self.generate_batch)
        return dataloader


def get_tokenizer(tokenizer_name):
    if "bert" in tokenizer_name == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        print("Bert Tokenizer Loaded!")
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
        print("Roberta Tokenizer Loaded!")
    return tokenizer


def train(model, iterator, optimizer):
    model.train()
    ep_t_loss, batch_num = 0, 0

    optimizer.zero_grad()
    for ix, batch in tqdm(enumerate(iterator)):
        inpt_ids, typ_ids, attn_masks, nsp_lbls, mlm_lbls = batch

        outputs = model(inpt_ids, attention_mask=attn_masks, token_type_ids=typ_ids,
                        next_sentence_label=nsp_lbls, labels=mlm_lbls)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        ep_t_loss += loss.item()
        batch_num += 1

    return ep_t_loss / batch_num


def test(model, iterator):
    model.eval()
    ep_t_loss, batch_num = 0, 0

    for ix, batch in tqdm(enumerate(iterator)):
        inpt_ids, typ_ids, attn_masks, nsp_lbls, mlm_lbls = batch

        with torch.no_grad():
            outputs = model(inpt_ids, attention_mask=attn_masks, token_type_ids=typ_ids,
                            next_sentence_label=nsp_lbls, labels=mlm_lbls)
        loss = outputs.loss
        ep_t_loss += loss.item()
        batch_num += 1
    return ep_t_loss / batch_num


def reduce_text_dict(dct, k=50000):
    test_dict = {"input_ids": dct["input_ids"][:k], "token_type_ids": dct["token_type_ids"][:k],
                 "attention_mask": dct["attention_mask"][:k], "nsp_labels": dct["nsp_labels"][:k]}

    train_dict = {"input_ids": dct["input_ids"][k:], "token_type_ids": dct["token_type_ids"][k:],
                  "attention_mask": dct["attention_mask"][k:], "nsp_labels": dct["nsp_labels"][k:]}

    return test_dict, train_dict


def main():
    base_transformer = "bert-base-uncased"
    use_gpu = "true"
    dataset = "imho"
    pre_processed_file_name = "/home/argumentation/reddit-argument-parser/data/imho+context-" + base_transformer + "_tokenized_dict.pkl"
    batch_size, num_workers = 64, 4
    num_epoch, start_epoch = 2, 0
    model_path = "/home/argumentation/reddit-argument-parser/results/model_backups/"
    model_name = model_path + base_transformer+"_imho_pretraining_AMP_replication.pt"
    file_start_index, ckpt_model = 1, None
    ckpt_model_best_loss = None

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_transformer", type=str, help="Pretrained transformer.", default=base_transformer)
    parser.add_argument("--pre_processed_file_name", type=str, help="pre_processed_file_name",
                        default=pre_processed_file_name)
    parser.add_argument("--use_gpu", type=str, help="use_gpu", default=use_gpu)
    parser.add_argument("--model_name", type=str, help="model_name", default=model_name)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=batch_size)
    parser.add_argument("--num_workers", type=int, help="num_workers", default=num_workers)
    parser.add_argument("--num_epoch", type=int, help="num_epoch", default=num_epoch)
    parser.add_argument("--start_epoch", type=int, help="start_epoch", default=start_epoch)
    parser.add_argument("--file_start_index", type=int, help="file_start_index", default=file_start_index)
    parser.add_argument("--ckpt_model", type=str, help="ckpt_model", default=ckpt_model)
    parser.add_argument("--ckpt_model_best_loss", type=float, help="ckpt_model_best_loss", default=ckpt_model_best_loss)
    parser.add_argument("--dataset", type=str, help="dataset", default=dataset)

    argv = parser.parse_args()
    base_transformer = argv.base_transformer
    pre_processed_file_name = argv.pre_processed_file_name
    use_gpu = True if argv.use_gpu == "true" else False
    batch_size = argv.batch_size
    num_workers = argv.num_workers
    num_epoch = argv.num_epoch
    file_start_index = argv.file_start_index
    ckpt_model = argv.ckpt_model
    ckpt_model_best_loss = argv.ckpt_model_best_loss
    start_epoch = argv.start_epoch
    model_name = argv.model_name
    dataset = argv.dataset

    tokenizer = get_tokenizer(base_transformer)

    print("\n:::: Starting Model Training ::::\n")
    device_num = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda:{}".format(device_num)) if torch.cuda.is_available() and use_gpu else "cpu"

    torch.distributed.init_process_group(backend="nccl")
    data_collator = DataCollator(tokenizer, device, batch_size, num_workers)

    model = BertForPreTraining.from_pretrained(base_transformer)

    if "bert" not in base_transformer:
        raise Exception("Cant' Pre-Train other models now!")

    ddp_model = model.to(device)
    if use_gpu and num_workers >= 2:
        ddp_model = torch.nn.parallel.DistributedDataParallel(ddp_model, device_ids=[device_num],
                                                              output_device=device_num)#, find_unused_parameters=True)

    if ckpt_model is not None:
        state_dict = torch.load(ckpt_model)
        ddp_model.load_state_dict(state_dict)
        print("\nModel Weights loaded from Checkpoint!\n")

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    best_valid_loss = ckpt_model_best_loss if ckpt_model_best_loss is not None else float('inf')
    early_stopping_marker = []
    if dataset == "imho":
        fname_test = pre_processed_file_name.replace(".pkl", "_shard_7.pkl")
        tokenized_dict_test = pickle.load(open(fname_test, "rb"))
        tokenized_dict_test_reduced, tokenized_dict_train_additional = reduce_text_dict(tokenized_dict_test)
        test_dataloader = data_collator.get_dataloader(tokenized_dict_test_reduced, training=False)

        for epoch in range(start_epoch, start_epoch+num_epoch):
            print("Epoch: {}, Training ...".format(epoch))

            for ix in range(file_start_index, 8):
                if ix == 7:
                    print("Training on Additional Training Data!")
                    tokenized_dict = copy.deepcopy(tokenized_dict_train_additional)
                else:
                    fname = pre_processed_file_name.replace(".pkl", "_shard_"+str(ix)+".pkl")
                    print("Training on File:", fname)
                    tokenized_dict = pickle.load(open(fname, "rb"))
                train_dataloader = data_collator.get_dataloader(tokenized_dict)
                train_loss = train(ddp_model, train_dataloader, optimizer)

                if device_num == 0:
                    print("Saving Model!")
                    model_name_nw = model_name.replace(".pt", "_epoch_" + str(epoch) + "_ckpt_" + str(ix) + ".pt")
                    torch.save(ddp_model.state_dict(), model_name_nw)

                    test_loss = test(ddp_model, test_dataloader)
                    if test_loss < best_valid_loss:
                        best_valid_loss = test_loss
                        print("\n::::: Epoch:", epoch, "Shard:", ix, "has best model!! :::::\n")
                        # model_name = model_name.replace(".pt", "_epoch_" + str(epoch) + "_ckpt_" + str(ix) + ".pt")
                        # torch.save(ddp_model.state_dict(), model_name)
                        early_stopping_marker.append(False)
                    else:
                        early_stopping_marker.append(True)

                    print("\n:::::::::::::::::::::::::::::::::::::")
                    print("Training Loss:", round(train_loss, 5))
                    print("Test Loss:", round(test_loss, 5))
                    print("=====================================\n")

                    if all(early_stopping_marker[-2:]):
                        print("Early stopping training as the Validation loss did NOT improve for last " + str(2) + \
                              " iterations.")
                        break
    else:
        tokenized_dict = pickle.load(open(pre_processed_file_name, "rb"))
        test_dict, train_dict = reduce_text_dict(tokenized_dict, k=10000)

        train_dataloader = data_collator.get_dataloader(train_dict)
        test_dataloader = data_collator.get_dataloader(test_dict, training=False)

        for epoch in range(start_epoch, start_epoch + num_epoch):
            print("Epoch: {}, Training ...".format(epoch))
            train_loss = train(ddp_model, train_dataloader, optimizer)
            if device_num == 0:
                print("Saving Model!")
                model_name_nw = model_name.replace(".pt", "_epoch_" + str(epoch) + ".pt")
                torch.save(ddp_model.state_dict(), model_name_nw)

                test_loss = test(ddp_model, test_dataloader)
                if test_loss < best_valid_loss:
                    best_valid_loss = test_loss
                    print("\n::::: Epoch:", epoch, "has best model!! :::::\n")
                    early_stopping_marker.append(False)
                else:
                    early_stopping_marker.append(True)

                print("\n:::::::::::::::::::::::::::::::::::::")
                print("Training Loss:", round(train_loss, 5))
                print("Test Loss:", round(test_loss, 5))
                print("=====================================\n")

                if all(early_stopping_marker[-2:]):
                    print("Early stopping training as the Validation loss did NOT improve for last " + str(2) + \
                          " iterations.")
                    break


if __name__ == '__main__':
    main()
