""" Trainer to Re-CREATE Ampersand results """

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
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import os
import json
import time
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Model(nn.Module):
    def __init__(self, base_model, task):
        super().__init__()
        self.base = base_model
        self.task = task
        if task == "component":
            self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        base_output = self.base(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.task == "component":
            hidden = torch.sum(base_output["hidden_states"][-1], 1)
            logits = self.classifier(hidden)
        else:
            logits = base_output["seq_relationship_logits"]

        res = {"prediction_logits": logits, "loss": None}
        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()
            pred = logits.contiguous().view(-1, logits.shape[-1])
            tgt = labels.contiguous().view(-1)
            loss = ce_loss(pred, tgt)
            res["loss"] = loss

        return res


class Trainer:
    def __init__(self, configuration, train_dict, test_dict):
        self.configuration = configuration
        self.device = torch.device("cuda:{}".format(configuration["device_num"])) if torch.cuda.is_available() and \
                                                                                     configuration["use_gpu"] else "cpu"
        self.model = self.load_model()
        print("\n::: Model Loaded :::\n")

        self.train_dataset, self.test_dataset = train_dict, test_dict
        self.train_dataloader, self.test_dataloader = self.get_dataloaders()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def load_model(self):
        base_model = BertForPreTraining.from_pretrained(self.configuration["base_transformer"],
                                                        output_hidden_states=True)
        if self.configuration["task"] in ["component"]:
            ignore_list = ["bert.pooler.dense.weight", "bert.pooler.dense.bias", "cls.predictions.bias",
                           "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias",
                           "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias",
                           "cls.seq_relationship.weight", "cls.seq_relationship.bias"]
        else:
            ignore_list = ["cls.predictions.bias",
                           "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias",
                           "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias"]

        ignore_regex_pattern = "("+"|".join([r"\b" + i + r"\b" for i in ignore_list])+")"
        grad_off_counter = 0
        for name, param in base_model.named_parameters():
            if len(re.findall(ignore_regex_pattern, name)) > 0:
                param.requires_grad = False
                grad_off_counter += 1
        print("\n", grad_off_counter, "parameter Grads turned off.\n")

        if self.configuration["pretrained_weights"] is not None:
            state_dict = torch.load(self.configuration["pretrained_weights"])
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            base_model.load_state_dict(state_dict)
            print("Model Parameters Loaded from Pre-Trained Checkpoint.\n")

        model = Model(base_model, self.configuration["task"])
        ddp_model = model.to(self.device)

        if self.configuration["use_gpu"] and self.configuration["num_workers"] >= 2:
            ddp_model = torch.nn.parallel.DistributedDataParallel(ddp_model,
                                                                  device_ids=[self.configuration["device_num"]],
                                                                  output_device=self.configuration["device_num"])
        return ddp_model

    def data_process(self, data_dict):
        data_lst = []
        for ix, i in enumerate(data_dict["input_ids"]):
            data_lst.append([i, data_dict["token_type_ids"][ix], data_dict["attention_mask"][ix],
                             data_dict["labels"][ix]])
        return data_lst

    def get_dataloaders(self):
        train_dataset_processed = self.data_process(self.train_dataset)
        test_dataset_processed = self.data_process(self.test_dataset)

        train_dataloader = DataLoader(train_dataset_processed, batch_size=self.configuration["batch_size"],
                                      sampler=DistributedSampler(train_dataset_processed),
                                      num_workers=self.configuration["num_workers"])
        test_dataloader = DataLoader(test_dataset_processed, batch_size=self.configuration["batch_size"],
                                     sampler=SequentialSampler(test_dataset_processed),
                                     num_workers=self.configuration["num_workers"])
        return train_dataloader, test_dataloader

    def classification_stats_suite(self, tgt_t, src_t, typ, ignore_val=-1):
        tgt, src = [], []
        for ix, i in enumerate(tgt_t):
            if i != ignore_val:
                tgt.append(i)
                src.append(src_t[ix])
        assert len(tgt) == len(src)
        cm = confusion_matrix(tgt, src)
        cs = classification_report(tgt, src)
        cs_dict = classification_report(tgt, src, zero_division=0, output_dict=True)
        print("\n===== STATS FOR ", typ, "=====")
        print("Confusion metric : \n", cm)
        print("Classification Stats:\n", cs)
        print("==============================\n")

        stat_lst = []  # Task, Label, Metric, Score
        for k, v in cs_dict.items():
            if k == 'accuracy':
                stat_lst.append([typ, "overall", k, v])
            else:
                stat_lst.append([typ, k, "f1-score", v["f1-score"]])
                stat_lst.append([typ, k, "precision", v["precision"]])
                stat_lst.append([typ, k, "recall", v["recall"]])
                stat_lst.append([typ, k, "support", v["support"]])
        return stat_lst

    def train(self):
        self.model.train()
        ep_t_loss, batch_num = 0, 0

        self.optimizer.zero_grad()
        for ix, batch in tqdm(enumerate(self.train_dataloader)):
            input_ids, type_ids, attention_masks, labels = batch
            outputs = self.model(input_ids, token_type_ids=type_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(name)
            self.optimizer.step()
            self.optimizer.zero_grad()
            ep_t_loss += loss.item()
            batch_num += 1
        return ep_t_loss / batch_num

    def test(self):
        self.model.eval()
        ep_t_loss, batch_num = 0, 0
        pred, tgt = [], []

        self.optimizer.zero_grad()
        for ix, batch in tqdm(enumerate(self.test_dataloader)):
            input_ids, type_ids, attention_masks, labels = batch

            with torch.no_grad():
                outputs = self.model(input_ids, token_type_ids=type_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs["loss"]
            ep_t_loss += loss.item()
            batch_num += 1

            pred.extend(outputs["prediction_logits"].argmax(dim=-1).detach().view(-1).tolist())
            tgt.extend(labels.detach().cpu().view(-1).tolist())

        stats = self.classification_stats_suite(tgt, pred, self.configuration["task"])
        return ep_t_loss / batch_num, stats


def process_data(df, task):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if task in ["inter", "intra"]:
        sentence_a, sentence_b, labels = [], [], []
        for ix, row in df.iterrows():
            sentence_a.append(row["sentence_a"])
            sentence_b.append(row["sentence_b"])
            labels.append(int(row["label"]))
        inputs = tokenizer(sentence_a, sentence_b, max_length=200, truncation=True,
                           return_tensors="pt", padding=True)
    else:
        sentence, labels = list(df["text"]), list(df["label"])
        inputs = tokenizer(sentence, max_length=200, truncation=True,
                           return_tensors="pt", padding=True)
    dct = {"input_ids": inputs.input_ids, "token_type_ids": inputs.token_type_ids,
           "attention_mask": inputs.attention_mask, "labels": torch.tensor(labels)}
    return dct


def load_data(task):
    path = "/home/argumentation/AMPERSAND-EMNLP2019/"
    file_paths = {"component": {"train": path+"claimtrain.tsv", "test": path+"claimdev.tsv",
                                "columns":["text", "label"]},
                  "intra": {"train": path + "intratrain.tsv", "test": path + "intradev.tsv",
                                "columns":["sentence_a", "sentence_b", "label"]},
                  "inter": {"train": path + "intertrain.tsv", "test": path + "interdev.tsv",
                                "columns":["sentence_a", "sentence_b", "label", "junk_id"]}
                  }
    train_df = pd.read_csv(file_paths[task]["train"], sep="\t", header=None)
    test_df = pd.read_csv(file_paths[task]["test"], sep="\t", header=None)
    train_df.columns = file_paths[task]["columns"]
    test_df.columns = file_paths[task]["columns"]

    train_dict, test_dict = process_data(train_df, task), process_data(test_df, task)
    return train_dict, test_dict


def main():
    configuration = {
        "base_transformer": "bert-base-uncased",
        "pretrained_weights": None,
        "task": "component",
        "device_num": 0,
        "use_gpu": True,
        "num_workers": 4,
        "batch_size": 64,
        "num_epochs": 15,
        "early_stopping": 2
    }
    configuration["output_model_name"] = configuration["base_transformer"] + "_" + configuration["task"] + "_model.pt"

    if configuration["task"] in ["component", "inter_relation"]:
        configuration["pretrained_weights"] = '<file_name.pt>'
    else:
        configuration["pretrained_weights"] = '<file_name.pt>'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_weights", type=str, help="pretrained_weights", default=configuration["pretrained_weights"])
    parser.add_argument("--task", type=str, help="task", default=configuration["task"])
    parser.add_argument("--batch_size", type=int, help="batch_size", default=configuration["batch_size"])
    parser.add_argument("--output_model_name", type=str, help="output_model_name", default=configuration["output_model_name"])

    argv = parser.parse_args()
    configuration["pretrained_weights"] = argv.pretrained_weights
    configuration["task"] = argv.task
    configuration["batch_size"] = argv.batch_size
    configuration["output_model_name"] = argv.output_model_name
    configuration["output_stats_name"] = configuration["output_model_name"].replace(".pt", "_stats.log")

    train_dict, test_dict = load_data(configuration["task"])
    print("\n:::: Training & Testing Data Loaded ::::\n")

    configuration["device_num"] = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")

    trainer = Trainer(configuration, train_dict, test_dict)
    print("\n:::: Trainer Loaded. Will Start Training ::::\n")
    best_valid_loss = float('inf')
    early_stopping_marker, best_stats = [], None

    for epoch in range(configuration["num_epochs"]):
        print("Epoch: {}, Training ...".format(epoch))
        train_loss = trainer.train()

        if configuration["device_num"] == 0:
            test_loss, test_stats = trainer.test()
            if test_loss < best_valid_loss:
                best_valid_loss = test_loss
                print("\n::::: Epoch:", epoch, "has best model!! :::::\n")
                print("Saving Best Model")
                torch.save(trainer.model.state_dict(), configuration["output_model_name"])
                early_stopping_marker.append(False)
                best_stats = test_stats
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

    if configuration["device_num"] == 0:
        print("Saving Execution STATS to:", configuration["output_stats_name"], "\n")
        with open(configuration["output_stats_name"], 'w') as fp:
            json.dump(best_stats, fp)
        print("ALL DONE!! SLEEPING & GOING TO EXIT!!")
        time.sleep(45)
        os._exit(1)


if __name__ == '__main__':
    main()