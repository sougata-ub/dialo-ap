""" The primary trainer for training our model """

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import argparse
import time
import tqdm
import math
import pickle
import json
import shutil
import pandas as pd
import os
import numpy as np
import copy
import random
import os.path
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
from datetime import datetime
from models import Parser
# from custom_dataloader import CustomIterableDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import re
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import utils

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


class Trainer:
    def __init__(self, configuration):
        self.set_random_seeds(random_seed=42)
        self.label_mapping = {'0': 0, 'Append': 3, 'support': 1, 'agreement': 1, 'partial_agreement': 1,
                              'understand': 1, 'rebuttal_attack': 2, 'partial_attack': 2, 'rebuttal': 2,
                              'undercutter_attack': 2, 'partial_disagreement': 2, 'disagreement': 2, 'undercutter': 2,
                              'attack': 2}
        self.segment_tags = {"B": 1, "O": 0}
        self.type_tags = {"non_arg": 0, "main_claim": 1, "claim": 2, "premise": 3}
        self.intent_tags = {'agreement': 1, 'disagreement': 2, 'ethos': 3, 'ethos_logos': 3, 'ethos_logos_pathos': 3,
                            'ethos_pathos': 3, 'evaluation_emotional': 6, 'evaluation_rational': 7, 'interpretation': 8,
                            'logos': 4, 'logos_pathos': 4, 'none': 0, 'pathos': 5}

        self.label_mapping_rev = {0: "no_rel", 1: "support", 2: "attack", 3: "append"}
        self.segment_tags_rev = {v: k for k, v in self.segment_tags.items()}
        self.type_tags_rev = {v: k for k, v in self.type_tags.items()}
        self.intent_tags_rev = {v: k for k, v in self.intent_tags.items()}

        self.device = torch.device("cuda:{}".format(configuration["device_num"])) if torch.cuda.is_available() and \
                                                                                     configuration["use_gpu"] else "cpu"
        self.configuration = configuration

        self.model, self.tokenizer, self.special_token_idx = self.get_model_v2()
        print("Model Loaded")

        if self.configuration["train_distributed"] and self.configuration["use_gpu"]:
            self.parser = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                    device_ids=[configuration["device_num"]],
                                                                    output_device=configuration["device_num"],
                                                                    find_unused_parameters=True)
            print("Distributed Model Loaded")
        else:
            self.parser = copy.deepcopy(self.model)
        print("Model has", self.count_parameters(), "parameters !")

        self.optimizer = torch.optim.AdamW(self.parser.parameters(), lr=configuration["lr"])
        self.scaler = GradScaler()

        if not configuration["pre_train"]:
            self.data_dict = pickle.load(open(configuration["training_file"], "rb"))
            self.curriculum_data_loader_list = self.get_dataloader_list()

    def set_random_seeds(self, random_seed=0):
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    def get_model_v2(self):
        tokenizer = RobertaTokenizerFast.from_pretrained(self.configuration["base_transformer"])
        config = RobertaConfig.from_pretrained(self.configuration["base_transformer"])

        keys_to_add = ["<EDU>"]
        special_tokens_dict = {'additional_special_tokens': keys_to_add}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        special_token_idx = [tokenizer.get_vocab()[i] for i in keys_to_add]
        print(num_added_toks, "tokens added\n")
        config.output_hidden_states = True

        base_model = RobertaModel.from_pretrained(self.configuration["base_transformer"], config=config)
        base_model.resize_token_embeddings(len(tokenizer))
        model = Parser(base_model, self.configuration)#.to(self.device)

        if self.configuration["increase_token_type"]:
            model.input_encoder.enc.config.type_vocab_size = 2
            model.input_encoder.enc.embeddings.token_type_embeddings = nn.Embedding(2,
                                                                                  model.input_encoder.enc.config.hidden_size)
            model.input_encoder.enc.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0,
                                                                                       std=model.input_encoder.enc.config.initializer_range)
            print("\n:::::: TOKEN TYPE EMBEDDINGS INCREASED ::::::\n")

        if self.configuration["increase_positional_embeddings"]:
            model.input_encoder.enc.config.max_position_embeddings = 2500
            old_position_embeddings_weight = model.input_encoder.enc.embeddings.position_embeddings.weight.data
            model.input_encoder.enc.embeddings.position_embeddings = nn.Embedding(
                model.input_encoder.enc.config.max_position_embeddings,
                model.input_encoder.enc.config.hidden_size)
            model.input_encoder.enc.embeddings.position_embeddings.weight.data.normal_(mean=0.0,
                                                                                     std=model.input_encoder.enc.config.initializer_range)
            model.input_encoder.enc.embeddings.position_embeddings.weight.data = torch.cat(
                [old_position_embeddings_weight,
                 model.input_encoder.enc.embeddings.position_embeddings.weight.data[514:, :]], 0)
            assert round(model.input_encoder.enc.embeddings.position_embeddings.weight.data[:514, :].sum().item()) == \
                   round(old_position_embeddings_weight.sum().item())
            print("\n:::::: POSITIONAL EMBEDDINGS INCREASED ::::::\n")

        if self.configuration["use_pretrained_base"]:
            state_dict = torch.load(self.configuration["pretrained_model"], map_location='cpu')
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print("\n:::::: PRE-TRAINED MODEL WEIGHTS LOADED ::::::\n")

        if self.configuration["use_curriculum_base"]:
            state_dict = torch.load(self.configuration["curriculum_model"], map_location='cpu')
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print("\nEncountered exception while loading state dict:", e)
                print("Will load model without enforcing strict state_dict match!\n")
                model.load_state_dict(state_dict, strict=False)
            print("\n:::::: CURRICULUM MODEL WEIGHTS LOADED ::::::\n")

        model = model.to(self.device)
        return model, tokenizer, special_token_idx

    def get_dataloader_list(self):
        """ Run for any different combinations of curriculums.
            Example input :
            run_curriculum = "C_local_component:C_local_global_component:C_local_global_component_segment:C_target_dataset:combined",
            run_curriculum = "C_local_component:C_local_global_component"
            run_curriculum = "combined"
        """
        if self.configuration["n_rerun"] > 1 or self.configuration["randomize_target_data"]:
            target_train_dict, target_test_dict = self.get_target_splits(self.data_dict["C_target_dataset"]["TRAIN"],
                                                                         self.data_dict["C_target_dataset"]["TEST"])
        else:
            target_train_dict, target_test_dict = self.shuffle_dict(self.data_dict["C_target_dataset"]["TRAIN"]), \
                                                  self.data_dict["C_target_dataset"]["TEST"]

        got_curriculums = [cid for cid in self.configuration["run_curriculum"].split(":")]

        print("GOT THE FOLLOWING CURRICULUM ==>", got_curriculums)

        curriculum_list = []
        for curri in got_curriculums:
            if curri == "combined":
                train_dict = self.shuffle_dict({**self.data_dict["C_local_global_component"]["TRAIN"],
                                                **self.data_dict["C_local_global_component_segment"]["TRAIN"],
                                                **self.data_dict["C_local_component"]["TRAIN"],
                                                **target_train_dict,
                                                **self.data_dict["C_local_global_component"]["TEST"],
                                                **self.data_dict["C_local_global_component_segment"]["TEST"],
                                                **self.data_dict["C_local_component"]["TEST"]})
                new_model_name = self.configuration["model_name"].replace(".pt", "_c_combined.pt")
                curriculum_list.append(["C_combined", new_model_name,
                                        self.get_dataloader(train_dict, training=True,
                                                            distributed=self.configuration["train_distributed"],
                                                            batch_size=min(1, self.configuration["batch_size"])),
                                        self.get_dataloader(target_test_dict, training=False, distributed=False,
                                                            batch_size=min(1, self.configuration["batch_size"]))
                                        ])
            else:
                new_model_name = self.configuration["model_name"].replace(".pt", "_" + curri.lower() + ".pt")
                bs = utils.curriculum_prediction_mapping[curri]["batch_size"]
                if curri != "C_target_dataset":
                    curriculum_list.append([curri, new_model_name,
                                            self.get_dataloader(self.shuffle_dict(self.data_dict[curri]["TRAIN"]),
                                                                training=True,
                                                                distributed=self.configuration["train_distributed"],
                                                                batch_size=bs),
                                            self.get_dataloader(self.data_dict[curri]["TEST"], training=False,
                                                                distributed=False,
                                                                batch_size=bs)
                                            ])
                else:
                    curriculum_list.append([curri, new_model_name,
                                            self.get_dataloader(target_train_dict, training=True,
                                                                distributed=self.configuration["train_distributed"],
                                                                batch_size=bs),
                                            self.get_dataloader(target_test_dict, training=False, distributed=False,
                                                                batch_size=bs)
                                            ])
        assert len(curriculum_list) > 0
        return curriculum_list

    def get_target_splits(self, dict1, dict2):
        c3_combined_dict = {**dict1, **dict2}
        c3_all_ids = list(set([v["uniq_id"].split(":")[1] for k, v in c3_combined_dict.items()]))
        random.shuffle(c3_all_ids)
        c3_len = len(c3_all_ids)
        c3_train_ids, c3_test_ids = set(c3_all_ids[:int(c3_len * 0.9)]), set(c3_all_ids[int(c3_len * 0.9):])

        c3_train_dict, c3_test_dict = {}, {}
        for k, v in c3_combined_dict.items():
            if v["uniq_id"].split(":")[1] in c3_train_ids:
                c3_train_dict[k] = v
            else:
                c3_test_dict[k] = v
        return self.shuffle_dict(c3_train_dict), c3_test_dict

    def shuffle_dict(self, dct):
        kys = list(dct.keys())
        random.shuffle(kys)
        return {i: dct[i] for i in kys}

    def count_parameters(self):
        return sum(p.numel() for p in self.parser.parameters() if p.requires_grad)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

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

    def train(self, iterator):  #, local_rank, model_name):
        self.parser.train()
        ep_t_loss, local_loss, local_loss_app, global_loss, segment_loss, typ_loss, context_loss = 0, 0, 0, 0, 0, 0, 0
        local_lbl_loss, global_lbl_loss = 0, 0
        batch_num = 0

        self.optimizer.zero_grad()
        for ix, batch in tqdm(enumerate(iterator)):
            ids, input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask, ctx_type_id, \
                segment_labels, component_labels, local_rel, local_rel_app, local_rel_labels, global_rel, \
                    global_rel_labels, global_ctx_adu_labels = batch

            with autocast():
                output_dct = self.parser(input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids,
                                         ctx_attention_mask, ctx_type_id, segment_labels, component_labels, local_rel,
                                         local_rel_labels, local_rel_app, global_rel, global_rel_labels,
                                         global_ctx_adu_labels)

                loss = 0.0 #torch.tensor(0.0) # alpha * output_dct["loss_local_rel_head"]
                for k, v in output_dct.items():
                    # if v is not None and k != "loss_local_rel_head" and k.startswith("loss"):
                    if v is not None and k.startswith("loss"):
                        loss += v * self.configuration[k+"_wt"] #(alpha if k in alpha_list else beta)

                loss = loss / self.configuration["accumulation"]

            self.scaler.scale(loss).backward()

            if (ix + 1) % self.configuration["accumulation"] == 0 or ix + 1 == len(iterator):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parser.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            ep_t_loss += loss.item()
            if output_dct.get("loss_local_rel_head", None) is not None:
                local_loss += output_dct["loss_local_rel_head"].item()

            if output_dct.get("loss_local_rel_head_app", None) is not None:
                local_loss_app += output_dct["loss_local_rel_head_app"].item()

            if output_dct.get("loss_global_rel_head", None) is not None:
                global_loss += output_dct["loss_global_rel_head"].item()

            if output_dct.get("loss_local_rel_deprel", None) is not None:
                local_lbl_loss += output_dct["loss_local_rel_deprel"].item()

            if output_dct.get("loss_global_rel_deprel", None) is not None:
                global_lbl_loss += output_dct["loss_global_rel_deprel"].item()

            if output_dct.get("loss_segmentation", None) is not None:
                segment_loss += output_dct["loss_segmentation"].item()

            if output_dct.get("loss_edu_type", None) is not None:
                typ_loss += output_dct["loss_edu_type"].item()

            if output_dct.get("loss_ctx_rel_deprel", None) is not None:
                context_loss += output_dct["loss_ctx_rel_deprel"].item()

            batch_num += 1

        return ep_t_loss/batch_num, local_loss/batch_num, local_loss_app/batch_num, global_loss/batch_num, \
               local_lbl_loss/batch_num, global_lbl_loss/batch_num, segment_loss/batch_num, typ_loss/batch_num, \
               context_loss/batch_num

    def evaluate(self, iterator):
        self.parser.eval()
        ep_t_loss, local_loss, local_loss_app, global_loss, segment_loss, typ_loss, context_loss = 0, 0, 0, 0, 0, 0, 0
        local_lbl_loss, global_lbl_loss = 0, 0
        local_src, local_tgt, global_src, global_tgt = [], [], [], []
        local_src_app, local_tgt_app = [], []
        local_src_lbl, local_tgt_lbl, global_src_lbl, global_tgt_lbl = [], [], [], []
        segment_src, segment_tgt = [], []
        typ_src, typ_tgt = [], []
        ctx_rel_src, ctx_rel_tgt = [], []
        batch_num = 0

        for ix, batch in tqdm(enumerate(iterator)):
            ids, input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask, ctx_type_id, \
                segment_labels, component_labels, local_rel, local_rel_app, local_rel_labels, global_rel, \
                    global_rel_labels, global_ctx_adu_labels = batch

            with autocast():
                with torch.no_grad():
                    output_dct = self.parser(input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask,
                                             ctx_type_id, segment_labels, component_labels, local_rel, local_rel_labels,
                                             local_rel_app, global_rel, global_rel_labels, global_ctx_adu_labels)

                loss = 0.0#torch.tensor(0.0)
                for k, v in output_dct.items():
                    if v is not None and k.startswith("loss"):
                        loss += v * self.configuration[k+"_wt"]

            ep_t_loss += loss.item()
            if output_dct.get("loss_local_rel_head", None) is not None:
                local_loss += output_dct["loss_local_rel_head"].item()

            if output_dct.get("loss_local_rel_head_app", None) is not None:
                local_loss_app += output_dct["loss_local_rel_head_app"].item()

            if output_dct.get("loss_global_rel_head", None) is not None:
                global_loss += output_dct["loss_global_rel_head"].item()

            if output_dct.get("loss_local_rel_deprel", None) is not None:
                local_lbl_loss += output_dct["loss_local_rel_deprel"].item()

            if output_dct.get("loss_global_rel_deprel", None) is not None:
                global_lbl_loss += output_dct["loss_global_rel_deprel"].item()

            if output_dct.get("loss_segmentation", None) is not None:
                segment_loss += output_dct["loss_segmentation"].item()

            if output_dct.get("loss_edu_type", None) is not None:
                typ_loss += output_dct["loss_edu_type"].item()

            if output_dct.get("loss_ctx_rel_deprel", None) is not None:
                context_loss += output_dct["loss_ctx_rel_deprel"].item()

            batch_num += 1

            if output_dct.get("logits_local_rel_head", None) is not None:
                local_src.extend((torch.sigmoid(output_dct["logits_local_rel_head"]) >=
                                  self.configuration["sigmoid_threshold"]).long().detach().view(-1).tolist())
                local_tgt.extend(local_rel.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_local_rel_head_app", None) is not None:
                local_src_app.extend((torch.sigmoid(output_dct["logits_local_rel_head_app"]) >=
                                      self.configuration["sigmoid_threshold"]).long().detach().view(-1).tolist())
                local_tgt_app.extend(local_rel_app.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_local_rel_deprel", None) is not None:
                local_src_lbl.extend(output_dct["logits_local_rel_deprel"].argmax(dim=-1).detach().view(-1).tolist())
                local_tgt_lbl.extend(local_rel_labels.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_global_rel_head", None) is not None:
                global_src.extend((torch.sigmoid(output_dct["logits_global_rel_head"]) >=
                                   self.configuration["sigmoid_threshold"]).long().detach().view(-1).tolist())
                # global_src.extend(output_dct["logits_global_rel_head"].argmax(dim=-1).detach().view(-1).tolist())
                global_tgt.extend(global_rel.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_global_rel_deprel", None) is not None:
                global_src_lbl.extend(output_dct["logits_global_rel_deprel"].argmax(dim=-1).detach().view(-1).tolist())
                global_tgt_lbl.extend(global_rel_labels.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_segmentation", None) is not None:
                segment_src.extend(output_dct["logits_segmentation"].argmax(dim=-1).detach().view(-1).tolist())
                segment_tgt.extend(segment_labels.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_edu_type", None) is not None:
                typ_src.extend(output_dct["logits_edu_type"].argmax(dim=-1).detach().view(-1).tolist())
                typ_tgt.extend(component_labels.detach().cpu().view(-1).tolist())

            if output_dct.get("logits_ctx_rel_deprel", None) is not None:
                ctx_rel_src.extend(output_dct["logits_ctx_rel_deprel"].argmax(dim=-1).detach().view(-1).tolist())
                ctx_rel_tgt.extend(global_ctx_adu_labels.detach().cpu().view(-1).tolist())

        stat_list = []
        if len(local_tgt) > 0:
            stats = self.classification_stats_suite(local_tgt, local_src, "Local Head Prediction", ignore_val=-1)
            stat_list.extend(stats)

        if len(local_tgt_app) > 0:
            stats = self.classification_stats_suite(local_tgt_app, local_src_app, "Local Head APPEND Prediction", ignore_val=-1)
            stat_list.extend(stats)

        if len(local_tgt_lbl) > 0:
            stats = self.classification_stats_suite(local_tgt_lbl, local_src_lbl, "Local Label Prediction", ignore_val=0)
            stat_list.extend(stats)

        if len(global_src) > 0:
            stats = self.classification_stats_suite(global_tgt, global_src, "Global Head Prediction", ignore_val=-1)
            stat_list.extend(stats)
            stats = self.classification_stats_suite(global_tgt_lbl, global_src_lbl, "Global Label Prediction", ignore_val=0)
            stat_list.extend(stats)

        if len(typ_tgt) > 0:
            stats = self.classification_stats_suite(typ_tgt, typ_src, "ADU Type Classification", ignore_val=-1)
            stat_list.extend(stats)

        if len(ctx_rel_tgt) > 0:
            stats = self.classification_stats_suite(ctx_rel_tgt, ctx_rel_src, "ADU Context Label Prediction", ignore_val=0)
            stat_list.extend(stats)

        if len(segment_tgt) > 0:
            stats = self.classification_stats_suite(segment_tgt, segment_src, "ADU Span Tagging", ignore_val=-1)
            stat_list.extend(stats)

        return ep_t_loss/batch_num, local_loss/batch_num, local_loss_app/batch_num, global_loss/batch_num, \
               local_lbl_loss/batch_num, global_lbl_loss/batch_num, segment_loss/batch_num, typ_loss/batch_num, \
               context_loss/batch_num, stat_list

    def get_discourse_pos(self, lst):
        return [ix for ix, i in enumerate(lst) if i in self.special_token_idx]

    def pad_sequence(self, pad, batch):
        maxlen = max([len(i) for i in batch])
        for ix in range(len(batch)):
            batch[ix].extend([pad] * (maxlen - len(batch[ix])))

    def get_splits(self, input_ids, threshold):
        idx_lst = [ix for ix, i in enumerate(input_ids) if i in self.special_token_idx]
        idx_lst.append(len(input_ids))
        lst = []
        counter, tmp = 0, (0, 0)

        for curr_pos, curr_ix in enumerate(idx_lst):
            if curr_pos != len(idx_lst) - 1:
                next_ix = idx_lst[curr_pos + 1]
                # If an edu itself is too long, truncate it
                if (next_ix - curr_ix) > threshold:
                    if tmp is not None:
                        lst.append(tmp)
                    lst.append((curr_ix, curr_ix + threshold))
                    counter, tmp = 0, (next_ix, next_ix)
                # If adding an edu crosses threshold, break at the edu
                elif (next_ix - curr_ix) + counter > threshold:
                    if tmp is not None:
                        lst.append(tmp)
                    counter, tmp = next_ix - curr_ix, (curr_ix, next_ix)
                # Else, continue appending positions
                else:
                    counter += (next_ix - curr_ix)
                    tmp = (tmp[0], next_ix)
            else:
                if tmp is not None:
                    lst.append(tmp)

        lst = [i for i in lst if i is not None and i[1] - i[0] > 0]
        return lst

    def split_input(self, input_ids, threshold=300):
        splits = self.get_splits(input_ids, threshold)
        input_ids_lst = []
        for i in splits:
            input_ids_lst.append(input_ids[i[0]: i[1]])
        return splits, input_ids_lst

    def data_process(self, data_dict):
        data_lst = []
        for k, v in data_dict.items():
            # if len(v["input_ids"]) <= 900: #and len(v["component_type"]) <= 100:
            pos_ids = []
            inpt_splits, inpt = self.split_input(v["input_ids"])

            prev_len = 2
            for inp in inpt:
                tmp_pos = list(range(prev_len, prev_len + len(inp)))
                prev_len += len(tmp_pos)
                pos_ids.append(tmp_pos)

            curr_pos = self.get_discourse_pos(v["input_ids"])

            edu_type = v["component_type"] if v["component_type"] is not None else None

            if edu_type is not None:
                assert len(curr_pos) == len(edu_type)

            ctx_input_ids = v["global_context_input_ids"] if v["global_context_input_ids"] is not None and \
                                                             len(v["global_context_input_ids"]) > 0 else None

            global_ctx_adu_labels = v["global_prev_adu_rel_mat"] if v["global_context_input_ids"] is not None and \
                                                                    len(v["global_context_input_ids"]) > 0 and v.get(
                "global_prev_adu_rel_mat", None) is not None \
                else None

            segment_labels = v["segmentation_tags"] if "segmentation_tags" in v.keys() \
                                                       and v["segmentation_tags"] is not None else None

            if segment_labels is not None:
                assert len([j for i in inpt for j in i]) == len(v["input_ids"]) == len(segment_labels)
            else:
                assert len([j for i in inpt for j in i]) == len(v["input_ids"])

            data_lst.append((v["uniq_id"], inpt, pos_ids, curr_pos, ctx_input_ids, segment_labels, edu_type,
                             v["local_rel_mat"], v["global_rel_mat"], global_ctx_adu_labels))
        return data_lst

    def get_single_data(self, dct):
        data_dict = copy.deepcopy(dct)
        data = []
        for k, v in data_dict.items():
            input_ids, attn_mask = [], []
            pos_ids = []
            type_ids = []
            inpt_splits, inpt = self.split_input(v["input_ids"])
            reconfig_input = [l for k in inpt_splits for l in v["input_ids"][k[0]:k[1]]]
            if "segmentation_tags" in v.keys() and v["segmentation_tags"] is not None:
                reconfig_segment_tags = [l for k in inpt_splits for l in v["segmentation_tags"][k[0]:k[1]]]
                segment_labels = torch.tensor(reconfig_segment_tags, dtype=torch.long).unsqueeze(0).to(self.device)
            else:
                segment_labels = None

            prev_len = 0
            for inp_ix, inp in enumerate(inpt):
                tmp_in = torch.tensor(inp, dtype=torch.long).unsqueeze(0).to(self.device)
                tmp_pos = torch.arange(prev_len, prev_len + tmp_in.shape[-1]).long().unsqueeze(0).to(self.device)
                prev_len += tmp_pos.shape[-1]

                input_ids.append(tmp_in)
                pos_ids.append(tmp_pos)

                attn_mask.append((tmp_in != self.tokenizer.pad_token_id).long().to(self.device))
                if v.get("curr_user_id", None) is not None:
                    type_ids.append((torch.ones_like(tmp_in) * int(v["curr_user_id"])).to(self.device))
                else:
                    type_ids = None

            curr_pos = torch.tensor(self.get_discourse_pos(reconfig_input), dtype=torch.long).unsqueeze(0).T

            if v["component_type"] is not None:
                edu_type = torch.tensor(v["component_type"], dtype=torch.long).unsqueeze(0).to(self.device)
            else:
                edu_type = None

            if v["global_context_input_ids"] is not None and len(v["global_context_input_ids"]) > 0:
                ctx_input_ids = v["global_context_input_ids"]
                self.pad_sequence(self.tokenizer.pad_token_id, ctx_input_ids)
                ctx_input_ids = torch.tensor(ctx_input_ids, dtype=torch.long).to(self.device)  # n_ctx, seq len
                ctx_attn_mask = (ctx_input_ids != self.tokenizer.pad_token_id).long().to(self.device)

                if v.get("ctx_user_id", None) is not None:
                    ctx_type_ids = torch.ones_like(ctx_input_ids) * torch.tensor(v["ctx_user_id"]).unsqueeze(-1).to(self.device)
                    ctx_type_ids = ctx_type_ids.to(self.device)
                else:
                    ctx_type_ids = None

                global_rel_mat_lbl = torch.tensor(v["global_rel_mat"], dtype=torch.long).unsqueeze(0).to(self.device)
                global_rel_mat = (global_rel_mat_lbl != 0).float().to(self.device)
                if v.get("global_prev_adu_rel_mat", None) is None:
                    global_ctx_adu_labels = None
                else:
                    global_ctx_adu_labels = torch.tensor(v["global_prev_adu_rel_mat"],
                                                         dtype=torch.long).unsqueeze(0).to(self.device)
            else:
                ctx_input_ids, ctx_attn_mask, global_rel_mat, ctx_type_ids = None, None, None, None
                global_rel_mat_lbl, global_ctx_adu_labels = None, None

            local_rel_mat_lbl = torch.tensor(v["local_rel_mat"], dtype=torch.long).unsqueeze(0).to(self.device)
            local_rel_mat = (local_rel_mat_lbl != 0).float().to(self.device)

            data.append((v["uniq_id"], input_ids, pos_ids, attn_mask, type_ids, curr_pos, ctx_input_ids, ctx_attn_mask, ctx_type_ids, \
                  segment_labels, edu_type, local_rel_mat, global_rel_mat, local_rel_mat_lbl, \
                  global_rel_mat_lbl, global_ctx_adu_labels))
        return data

    def pad_list_2d(self, batch, pad_token=None):
        pad_token = self.tokenizer.pad_token_id if pad_token is None else pad_token
        maxlen = max([len(i) for i in batch])
        padded_lst = []

        for i in batch:
            padded_lst.append(i + [pad_token] * (maxlen - len(i)))
        return torch.tensor(padded_lst)

    def pad_list_2_3dtensor(self, lst, pad_token=None):
        pad_token = self.tokenizer.pad_token_id if pad_token is None else pad_token

        max_len = max([len(j) for i in lst for j in i])
        max_splits = max([len(i) for i in lst])

        padded_lst = []
        for i in lst:
            tmp_lst = [j + [pad_token] * (max_len - len(j)) for j in i]
            tmp_lst.extend([[pad_token] * max_len] * (max_splits - len(tmp_lst)))
            padded_lst.append(tmp_lst)
        padded_tensor = torch.tensor(padded_lst)
        return padded_tensor

    def get_reformatted_cur_pos_idx(self, input_ids, pad_token):
        batch_size = input_ids.shape[0]
        x, y = torch.where(torch.isin(input_ids.view(batch_size, -1), torch.tensor([self.special_token_idx])))
        x, y = x.tolist(), y.tolist()

        idx_lst, tmp = [], []
        prev = None
        for ix, i in enumerate(x):
            if i != prev and prev is not None:
                idx_lst.append(tmp)
                tmp = [y[ix]]
            else:
                tmp.append(y[ix])
            prev = i
        if len(tmp) > 0:
            idx_lst.append(tmp)
        return self.pad_list_2d(idx_lst, pad_token)

    def get_reformatted_segment_labels(self, input_ids, segment_labels, pad_token):
        input_ids = input_ids.view(input_ids.shape[0], -1)
        final_segment_labels = torch.zeros_like(input_ids) + pad_token
        x, y = torch.where(input_ids != 1)

        loc_lst, tmp = [], []
        prev = None
        for i in zip(x, y):
            if i[0] != prev:
                if len(tmp) > 0:
                    loc_lst.append(tmp)
                tmp = [i]
            else:
                tmp.append(i)
            prev = i[0]
        if len(tmp) > 0:
            loc_lst.append(tmp)

        for i in loc_lst:
            for ix, j in enumerate(i):
                final_segment_labels[j] = segment_labels[j[0]][ix]
        return final_segment_labels

    def generate_batch(self, data_batch):
        uniq_ids, inpt_ids, pos_ids, curr_poss = [], [], [], []
        segment_labels, edu_types = [], []
        local_rel_mats, local_rel_mats_app, local_rel_mats_lbls = [], [], []
        global_rel_mats, global_rel_mats_lbls = [], []
        ctx_input_ids, global_ctx_adu_lbls = [], []

        for (uniq_id, inpt_id, pos_id, curr_pos, ctx_input_id, segment_label, edu_type,
             local_rel_mat, global_rel_mat, global_ctx_adu_label) in data_batch:
            inpt_ids.append(inpt_id)
            uniq_ids.append(uniq_id)
            pos_ids.append(pos_id)
            curr_poss.append(curr_pos)

            if ctx_input_id is not None:
                ctx_input_ids.append(ctx_input_id)
            else:
                ctx_input_ids.append([[self.tokenizer.pad_token_id]])

            if segment_label is not None:
                segment_labels.append(segment_label)
            else:
                len_inpt_ids = len([j for i in inpt_id for j in i])
                segment_labels.append([-1] * len_inpt_ids)

            if edu_type is not None:
                edu_types.append(edu_type)
            else:
                len_curr_pos = len(curr_pos)
                edu_types.append([-1] * len_curr_pos)

            if local_rel_mat is None:
                local_rel_mat = np.zeros((1, 1))
            app_mask = (local_rel_mat != self.label_mapping["Append"])
            local_rel_mats_lbls.append((local_rel_mat * app_mask).astype(int).tolist())
            local_rel_mats_app.append((local_rel_mat == self.label_mapping["Append"]).astype(float).tolist())
            local_rel_mats.append(((local_rel_mat * app_mask) != 0).astype(float).tolist())

            if global_rel_mat is not None:
                global_rel_mats_lbls.append(global_rel_mat.astype(int).tolist())
                global_rel_mats.append((global_rel_mat != 0).astype(float).tolist())
            else:
                global_rel_mats_lbls.append(np.zeros_like(local_rel_mat)[:1].astype(int).tolist())
                global_rel_mats.append((np.zeros_like(local_rel_mat)[:1]-1).astype(float).tolist())

            if global_ctx_adu_label is not None:
                global_ctx_adu_lbls.append(global_ctx_adu_label.astype(int).tolist())
            else:
                global_ctx_adu_lbls.append([[0]])

        inpt_ids_padded = self.pad_list_2_3dtensor(inpt_ids)  # batch, n_split, seq_len
        pos_ids_padded = self.pad_list_2_3dtensor(pos_ids)  # batch, n_split, seq_len
        attn_mask = inpt_ids_padded.ne(self.tokenizer.pad_token_id).int()  # batch, n_split, seq_len

        curr_poss_padded = self.get_reformatted_cur_pos_idx(inpt_ids_padded, pad_token=-1)
        curr_poss_padded = curr_poss_padded.transpose(0, 1)  # n, batch_size

        ctx_input_ids_padded = self.pad_list_2_3dtensor(ctx_input_ids)  # batch, n_ctx_turns, seq_len
        ctx_attn_mask = ctx_input_ids_padded.ne(self.tokenizer.pad_token_id).int()  # batch, n_ctx_turns, seq_len

        segment_labels_padded = self.get_reformatted_segment_labels(inpt_ids_padded, segment_labels, -1) #self.pad_list_2d(segment_labels, -1)
        edu_types_padded = self.pad_list_2d(edu_types, -1)

        local_rel_mats_lbls_padded = self.pad_list_2_3dtensor(local_rel_mats_lbls, 0)
        local_rel_mats_padded = self.pad_list_2_3dtensor(local_rel_mats, -1)
        local_rel_mats_app_padded = self.pad_list_2_3dtensor(local_rel_mats_app, -1)

        global_rel_mats_lbls_padded = self.pad_list_2_3dtensor(global_rel_mats_lbls, 0)
        global_rel_mats_padded = self.pad_list_2_3dtensor(global_rel_mats, -1)

        global_ctx_adu_lbls_padded = self.pad_list_2_3dtensor(global_ctx_adu_lbls, 0)

        type_ids, ctx_type_ids = None, None

        if ctx_attn_mask.sum().item() == 0:
            ctx_input_ids_padded, ctx_attn_mask = None, None

        if (segment_labels_padded != -1).sum().item() == 0:
            segment_labels_padded = None

        if (edu_types_padded != 0).sum().item() == 0:
            edu_types_padded = None

        if (local_rel_mats_padded != -1).sum().item() == 0:
            local_rel_mats_padded = None

        if (local_rel_mats_app_padded != -1).sum().item() == 0:
            local_rel_mats_app_padded = None

        if (local_rel_mats_lbls_padded != 0).sum().item() == 0:
            local_rel_mats_lbls_padded = None

        if (global_rel_mats_padded != -1).sum().item() == 0:
            global_rel_mats_padded = None

        if (global_rel_mats_lbls_padded != 0).sum().item() == 0:
            global_rel_mats_lbls_padded = None

        if (global_ctx_adu_lbls_padded != 0).sum().item() == 0:
            global_ctx_adu_lbls_padded = None

        assert inpt_ids_padded.shape == pos_ids_padded.shape == attn_mask.shape
        if type_ids is not None:
            assert type_ids.shape == inpt_ids_padded.shape

        return uniq_ids, inpt_ids_padded, pos_ids_padded, attn_mask, type_ids, curr_poss_padded, ctx_input_ids_padded, \
               ctx_attn_mask, ctx_type_ids, segment_labels_padded, edu_types_padded, local_rel_mats_padded, \
               local_rel_mats_app_padded, local_rel_mats_lbls_padded, global_rel_mats_padded, \
               global_rel_mats_lbls_padded, global_ctx_adu_lbls_padded

    def get_dataloader(self, encoded_dict, training=False, distributed=True, batch_size=8):
        data = self.data_process(encoded_dict)
        if self.configuration["n_examples"] is not None:
            data = data[:self.configuration["n_examples"]]
            print("Training with modified data size of", len(data), "examples.")
        else:
            print("Training with original data size of", len(data), "examples.")
        if training:
            if distributed:
                dataloader = DataLoader(data, batch_size=batch_size,
                                        sampler=DistributedSampler(data), collate_fn=self.generate_batch,
                                        num_workers=self.configuration["num_workers"])
            else:
                dataloader = DataLoader(data, batch_size=batch_size,
                                        sampler=RandomSampler(data), collate_fn=self.generate_batch,
                                        num_workers=self.configuration["num_workers"])
        else:
            dataloader = DataLoader(data, batch_size=batch_size,
                                    sampler=SequentialSampler(data), collate_fn=self.generate_batch,
                                    num_workers=self.configuration["num_workers"])
        return dataloader

    def get_pretrain_dataloader(self, filename, train=False):
        dataset = pickle.load(open(filename, "rb"))
        if train:
            if self.configuration["train_distributed"]:
                dataloader = DataLoader(dataset, batch_size=self.configuration["batch_size"],
                                        sampler=DistributedSampler(dataset),
                                        num_workers=self.configuration["num_workers"],
                                        collate_fn=self.generate_batch)
            else:
                dataloader = DataLoader(dataset, batch_size=self.configuration["batch_size"],
                                        sampler=RandomSampler(dataset),
                                        num_workers=self.configuration["num_workers"],
                                        collate_fn=self.generate_batch)
        else:
            dataloader = DataLoader(dataset, batch_size=self.configuration["batch_size"],
                                    sampler=SequentialSampler(dataset), num_workers=self.configuration["num_workers"],
                                    collate_fn=self.generate_batch)
        return dataloader
