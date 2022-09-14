""" E2E Parser built using our trained model """

import random
import pickle
import json
import logging
import pandas as pd
import os
import requests
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaModel
from models import Parser
from inference import Inference
import utils
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from collections import OrderedDict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class ArgParser(Inference):
    def __init__(self, configuration, model_name, evaluate=True, debug=True):
        super().__init__(configuration, model_name, evaluate)
        self.debug = debug

    def predict(self, edu_segments, context=None):
        segmented_text = utils.edu_to_text(edu_segments)
        if self.debug:
            print("\nSegmented Text with EDU token:", segmented_text, "\n")
        input_ids = self.tokenizer(segmented_text, add_special_tokens=False).input_ids
        context_input_ids = self.tokenizer(context, add_special_tokens=True).input_ids if context is not None else None

        data_dict = {"test": {"uniq_id": "inference", "input_ids": input_ids, "component_type": None,
                              "global_context_input_ids": context_input_ids, "global_prev_adu_rel_mat": None,
                              "segmentation_tags": None, "local_rel_mat": None, "global_rel_mat": None}
                     }

        data_batch = self.data_process(data_dict)
        if self.debug:
            print("\ndata_batch:::::\n", data_batch)
        ids, input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask, ctx_type_id, \
            segment_labels, component_labels, local_rel, local_rel_app, local_rel_labels, global_rel, \
                global_rel_labels, global_ctx_adu_labels = self.generate_batch(data_batch)

        input_ids, pos_ids, attention_mask, pos_arr = input_ids.to(self.device), pos_ids.to(self.device), \
                                                        attention_mask.to(self.device), pos_arr.to(self.device)
        if ctx_input_ids is not None:
            ctx_input_ids, ctx_attention_mask = ctx_input_ids.to(self.device), ctx_attention_mask.to(self.device)

        with autocast():
            with torch.no_grad():
                output_dct = self.parser.predict(input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids,
                                                 ctx_attention_mask, ctx_type_id)
                output_dct["input_ids"] = input_ids
                output_dct["uniq_id"] = ids
                output_dct["ctx_input_ids"] = ctx_input_ids
                output_dct = self.get_span_and_relations(output_dct)

        output_dct["segmented_text"] = segmented_text
        return output_dct

    def get_arg_components(self, op, keys_remap):
        component_dict = {}
        for k, v in op["pred_dict_new"].items():
            ky = keys_remap[k]
            txt = self.tokenizer.decode(v["tokens"], skip_special_tokens=False)
            txt = txt.replace("<EDU>", " ").replace("<pad>", "").strip()
            component_dict[ky] = {"text": txt, "type": v["label"]}
        return component_dict

    def get_intra_relations(self, op, keys_remap, component_dict):
        intra_relations = {}
        for ix, (k, v) in enumerate(op["pred_rels_map"].items()):
            if v["rel"] in [1, 2]:
                dest, source, rel = keys_remap[k[0]], keys_remap[k[1]], self.label_mapping_rev[v["rel"]]
                intra_relations[ix + 1] = {"parent_component": component_dict[dest],
                                           "child_component": component_dict[source],
                                           "relationship": rel}
        return intra_relations

    def get_inter_relations(self, op, id_lst, keys_remap, component_dict, component_dict_all):
        inter_relations = {}
        for k, v in op["global_txt_dict"].items():
            if v["rel"] in ["support", "attack"]:
                dest, source = id_lst[v["global_id"]], keys_remap[v["local_id"]]
                inter_relations[k + 1] = {"parent_component": component_dict_all[dest],
                                          "child_component": component_dict[source],
                                          "relationship": v["rel"]}
        return inter_relations

    def run(self, conversation):
        turns = [i.strip() for i in conversation.split("\t")]
        keys_remap = OrderedDict()
        turn_dict = OrderedDict()
        component_dict_all = OrderedDict()
        intra_relations_all = OrderedDict()
        inter_relations_all = OrderedDict()
        debug_output_dict = OrderedDict()
        intra_counter, inter_counter = 0, 0

        for ix, text in enumerate(turns):
            turn_id = "turn:"+str(ix+1)
            turn_dict[turn_id] = text
            edu_segments = utils.get_segmented_edus(text)

            if self.debug:
                print("\nTime taken to segment EDUs:", edu_segments["time_taken"], "\n")

            """ Getting Context """
            id_lst, context = [], []
            for comp_id, comp in component_dict_all.items():
                if comp["type"] in ["premise", "claim"]:
                    id_lst.append(comp_id)
                    context.append(comp["text"])
            if len(context) == 0:
                context = None

            """ Predict using model """
            output_dct = self.predict(edu_segments["edus"], context)
            output_dct["pred_dict_new"] = {k: output_dct["pred_dict_new"][k] for k in
                                           sorted(list(output_dct["pred_dict_new"].keys()))}
            keys_remap[turn_id] = {i: ix + 1 for ix, i in enumerate(sorted(list(output_dct["pred_dict_new"].keys())))}
            debug_output_dict[turn_id] = {"model_output": output_dct, "keys_remap": keys_remap[turn_id]}

            """ Get the argumentative components and add to the global component dict"""
            component_dict = self.get_arg_components(output_dct, keys_remap[turn_id])
            for k, v in component_dict.items():
                new_ky = turn_id + ":component:" + str(k)
                v["id"] = new_ky
                component_dict_all[new_ky] = v

            """ Get intra relations (local)"""
            intra_relations = self.get_intra_relations(output_dct, keys_remap[turn_id], component_dict)
            for k, v in intra_relations.items():
                intra_relations_all[intra_counter + k] = v
                intra_counter += 1

            """ Get inter relations (global)"""
            inter_relations = self.get_inter_relations(output_dct, id_lst, keys_remap[turn_id], component_dict,
                                                       component_dict_all)
            for k, v in inter_relations.items():
                inter_relations_all[inter_counter + k] = v
                inter_counter += 1

        res = {"turns": turn_dict, "argumentative_components": component_dict_all,
               "intra_relationships": intra_relations_all, "inter_relationships": inter_relations_all}

        if self.debug:
            res["debug_output"] = debug_output_dict

        return res

    def run_v2(self, turns):
        keys_remap = OrderedDict()
        turn_dict = OrderedDict()
        component_dict_all = OrderedDict()
        intra_relations_all = OrderedDict()
        inter_relations_all = OrderedDict()
        debug_output_dict = OrderedDict()
        intra_counter, inter_counter = 0, 0

        for ix, turn in enumerate(turns):
            user_id, text = turn
            turn_id = "turn:"+str(ix+1)+":user"+str(user_id)
            turn_dict[turn_id] = text
            edu_segments = utils.get_segmented_edus(text)

            if self.debug:
                print("\nTime taken to segment EDUs:", edu_segments["time_taken"], "\n")

            """ Getting Context """
            id_lst, context = [], []
            for comp_id, comp in component_dict_all.items():
                if comp["type"] in ["premise", "claim"] and comp_id.split(":")[3] != str(user_id):
                    id_lst.append(comp_id)
                    context.append(comp["text"])
            if len(context) == 0:
                context = None

            """ Predict using model """
            output_dct = self.predict(edu_segments["edus"], context)
            output_dct["pred_dict_new"] = {k: output_dct["pred_dict_new"][k] for k in
                                           sorted(list(output_dct["pred_dict_new"].keys()))}
            keys_remap[turn_id] = {i: ix + 1 for ix, i in enumerate(sorted(list(output_dct["pred_dict_new"].keys())))}
            debug_output_dict[turn_id] = {"model_output": output_dct, "keys_remap": keys_remap[turn_id]}

            """ Get the argumentative components and add to the global component dict"""
            component_dict = self.get_arg_components(output_dct, keys_remap[turn_id])
            for k, v in component_dict.items():
                new_ky = turn_id + ":component:" + str(k)
                v["id"] = new_ky
                component_dict_all[new_ky] = v

            """ Get intra relations (local)"""
            intra_relations = self.get_intra_relations(output_dct, keys_remap[turn_id], component_dict)
            for k, v in intra_relations.items():
                intra_relations_all[intra_counter + k] = v
                intra_counter += 1

            """ Get inter relations (global)"""
            inter_relations = self.get_inter_relations(output_dct, id_lst, keys_remap[turn_id], component_dict,
                                                       component_dict_all)
            for k, v in inter_relations.items():
                inter_relations_all[inter_counter + k] = v
                inter_counter += 1

        res = {"turns": turn_dict, "argumentative_components": component_dict_all,
               "intra_relationships": intra_relations_all, "inter_relationships": inter_relations_all}

        if self.debug:
            res["debug_output"] = debug_output_dict

        return res