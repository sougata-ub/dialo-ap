""" Class to perform Inference using our trained model. It is a subclass of the Trainer class,
    and a superclass of the ArgPArser class. """
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from datetime import datetime
from trainer import Trainer
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import utils
from torch.cuda.amp import autocast

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class Inference(Trainer):
    def __init__(self, configuration, model_name, evaluate=False):
        super().__init__(configuration)
        self.finetuned_model = model_name
        print("Initializing model weights from here:",model_name)
        state_dict = torch.load(model_name)
        state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
        self.parser.load_state_dict(state_dict)
        print("Model weights initialized!\n")

        self.parser.eval()
        self.evaluate = evaluate

    def run_inference(self):
        inference_dict = self.inference_predictor_v2()
        stat_list_df = pd.DataFrame(inference_dict["stat_list"],
                                    columns=["task", "metric", "class label", "num samples", "value"])
        stat_list_df["model"] = self.finetuned_model
        # stat_list_df.to_csv(self.stats_file, index=False)

        print("\n:::::STATISTICS:::::\n")
        print(stat_list_df)
        print("=====================================\n")
        # print("Dumped Stats results to file", self.stats_file, "!!\n")
        # pickle.dump(additional_data, open(self.inference_file, "wb"))
        # print("Dumped inference results to file", self.inference_file, "!!")

    def inference_predictor_v2(self):
        result_dict = {}
        for ix, batch in tqdm(enumerate(self.curriculum_data_loader_list[-1][-1])):
            ids, input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask, ctx_type_id, \
                segment_labels, component_labels, local_rel, local_rel_app, local_rel_labels, global_rel, \
                    global_rel_labels, global_ctx_adu_labels = batch

            with autocast():
                with torch.no_grad():
                    output_dct = self.parser(input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask,
                                             ctx_type_id, segment_labels, component_labels, local_rel, local_rel_labels,
                                             local_rel_app, global_rel, global_rel_labels, global_ctx_adu_labels)
                    output_dct["input_ids"] = input_ids
                    output_dct["uniq_id"] = ids
                    output_dct["local_rel"] = local_rel
                    output_dct["global_rel"] = global_rel
                    output_dct["local_rel_labels"] = local_rel_labels
                    output_dct["global_rel_labels"] = global_rel_labels
                    output_dct["ctx_input_ids"] = ctx_input_ids
                    output_dct["segment_labels"] = segment_labels
                    output_dct["component_labels"] = component_labels
                    output_dct = self.get_span_and_relations(output_dct)
                    result_dict[ix] = output_dct

    def inference_predictor(self):
        result_dict = {}
        test_dataloader = self.get_single_data(self.data_dict["C3"]["TEST"])
        for ix, batch in tqdm(enumerate(test_dataloader)):
            uniq_id, input_ids, pos_ids, attention_mask, type_id, pos_arr, ctx_input_ids, ctx_attention_mask, \
                ctx_type_id, segment_labels, component_labels, local_rel, global_rel, local_rel_labels, \
                    global_rel_labels, global_ctx_adu_labels = batch
            with torch.no_grad():
                output_dct = self.parser(input_ids.to(self.device), pos_ids.to(self.device), attention_mask.to(self.device), type_id,
                                         pos_arr.to(self.device), ctx_input_ids.to(self.device),
                                         ctx_attention_mask.to(self.device), ctx_type_id, segment_labels,
                                         component_labels, local_rel, local_rel_labels,
                                         global_rel, global_rel_labels, global_ctx_adu_labels)
                output_dct["input_ids"] = input_ids
                output_dct["uniq_id"] = uniq_id
                output_dct["local_rel"] = local_rel
                output_dct["global_rel"] = global_rel
                output_dct["local_rel_labels"] = local_rel_labels
                output_dct["global_rel_labels"] = global_rel_labels
                output_dct["ctx_input_ids"] = ctx_input_ids
                output_dct["segment_labels"] = segment_labels
                output_dct["component_labels"] = component_labels
                output_dct = self.get_span_and_relations(output_dct)
                result_dict[uniq_id] = output_dct
        stat_list, span_matches_df, rel_matches_df = self.process_inference_predictor(result_dict)
        return {"stat_list": stat_list, "span_matches_df": span_matches_df,
                "rel_matches_df": rel_matches_df, "result_dict": result_dict}

    def process_inference_predictor(self, result_dict):
        stat_list = []

        """ Segmentation Metrics """
        span_matches_df = pd.DataFrame([i for uniq_id, v in result_dict.items() for i in v["overlap_lst"]],
                                       columns=["uniq_id", "gold_adu_id", "pred_adu_id", "span_gold", "span_pred",
                                                "match_pct", "gold_lbl", "pred_lbl"])
        span_matches_df["partial_match"] = span_matches_df.apply(lambda x: utils.label_match(x["span_gold"],
                                                                                             x["span_pred"],
                                                                                             x["match_pct"], 0.5), 1)
        span_matches_df["full_match"] = span_matches_df.apply(lambda x: utils.label_match(x["span_gold"], x["span_pred"],
                                                                                          x["match_pct"], 1.0), 1)

        partial_dict = span_matches_df.partial_match.value_counts().to_dict()
        full_dict = span_matches_df.full_match.value_counts().to_dict()

        pm_f1, pm_pr, pm_rl, pm_n = utils.get_standard_metrics(partial_dict)
        fm_f1, fm_pr, fm_rl, fm_n = utils.get_standard_metrics(full_dict)

        stat_list.extend([["partial-match:span", "f1", "True", pm_n, pm_f1],
                          ["partial-match:span", "precision", "True", pm_n, pm_pr],
                          ["partial-match:span", "recall", "True", pm_n, pm_rl],
                          ["full-match:span", "f1", "True", fm_n, fm_f1],
                          ["full-match:span", "precision", "True", fm_n, fm_pr],
                          ["full-match:span", "recall", "True", fm_n, fm_rl]])

        """ Span Metrics """
        partial_match_df = span_matches_df[span_matches_df["partial_match"].isin(["tp", "tn"])]
        full_match_df = span_matches_df[span_matches_df["full_match"].isin(["tp", "tn"])]
        span_label_class_report_partial = classification_report(list(partial_match_df["gold_lbl"]),
                                                                list(partial_match_df["pred_lbl"]),
                                                                zero_division=0, output_dict=True)
        span_label_class_report_full = classification_report(list(full_match_df["gold_lbl"]),
                                                             list(full_match_df["pred_lbl"]),
                                                             zero_division=0, output_dict=True)

        dummy_mapping = {k: k for k, v in self.type_tags.items()}
        span_label_partial_report = utils.get_report_metrics("partial-match:span-label", span_label_class_report_partial,
                                                             dummy_mapping)
        span_label_full_report = utils.get_report_metrics("full-match:span-label", span_label_class_report_full,
                                                          dummy_mapping)
        stat_list.extend(span_label_partial_report + span_label_full_report)

        """ Relation Metrics  """
        rel_matches_df = pd.DataFrame([i for uniq_id, v in result_dict.items() for i in v["expected_mapping"]],
                                      columns=["uniq_id", "gold_tup", "expected_tup", "expected_tup_match_pct",
                                               "gold_tup_rel", "expected_tup_exists", "expected_tup_rel_exists",
                                               "expected_tup_rel"])
        rel_matches_df["freq"] = 1
        rel_matches_df = rel_matches_df.fillna(-1)

        rel_matches_df["expected_tup_exists_partial"] = rel_matches_df.apply(lambda x: \
                                                                                 utils.mark_false_negative(
                                                                                     x["expected_tup_exists"],
                                                                                     x["expected_tup_match_pct"], 0.5),
                                                                             1)
        rel_matches_df["expected_tup_rel_exists_partial"] = rel_matches_df.apply(lambda x: \
                                                                                     utils.mark_false_negative(
                                                                                         x["expected_tup_rel_exists"],
                                                                                         x["expected_tup_match_pct"],
                                                                                         0.5), 1)
        rel_matches_df["expected_tup_exists_full"] = rel_matches_df.apply(lambda x: \
                                                                              utils.mark_false_negative(
                                                                                  x["expected_tup_exists"],
                                                                                  x["expected_tup_match_pct"], 1.0), 1)
        rel_matches_df["expected_tup_rel_exists_full"] = rel_matches_df.apply(lambda x: \
                                                                                  utils.mark_false_negative(
                                                                                      x["expected_tup_rel_exists"],
                                                                                      x["expected_tup_match_pct"],
                                                                                      1.0), 1)
        rel_matches_df["expected_tup_rel_part"] = rel_matches_df.apply(lambda x: \
                                                                           utils.change_rel_label(
                                                                               x["expected_tup_rel"],
                                                                               x["expected_tup_match_pct"], 0.5), 1)
        rel_matches_df["expected_tup_rel_full"] = rel_matches_df.apply(lambda x: \
                                                                           utils.change_rel_label(
                                                                               x["expected_tup_rel"],
                                                                               x["expected_tup_match_pct"], 1.0), 1)
        partial_dict_rel = rel_matches_df.expected_tup_exists_partial.value_counts().to_dict()
        full_dict_rel = rel_matches_df.expected_tup_exists_full.value_counts().to_dict()
        partial_dict_rel_lbl = rel_matches_df.expected_tup_rel_exists_partial.value_counts().to_dict()
        full_dict_rel_lbl = rel_matches_df.expected_tup_rel_exists_full.value_counts().to_dict()

        prel_f1, prel_pr, prel_rl, prel_n = utils.get_standard_metrics(partial_dict_rel)
        frel_f1, frel_pr, frel_rl, frel_n = utils.get_standard_metrics(full_dict_rel)
        prel_lbl_f1, prel_lbl_pr, prel_lbl_rl, prel_lbl_n = utils.get_standard_metrics(partial_dict_rel_lbl)
        frel_lbl_f1, frel_lbl_pr, frel_lbl_rl, frel_lbl_n = utils.get_standard_metrics(full_dict_rel_lbl)

        lst = [["partial-match:relations", "f1", "True", prel_n, prel_f1],
               ["partial-match:relations", "precision", "True", prel_n, prel_pr],
               ["partial-match:relations", "recall", "True", prel_n, prel_rl],
               ["full-match:relations", "f1", "True", frel_n, frel_f1],
               ["full-match:relations", "precision", "True", frel_n, frel_pr],
               ["full-match:relations", "recall", "True", frel_n, frel_rl],
               ["partial-match:relations-label", "f1", "True", prel_lbl_n, prel_lbl_f1],
               ["partial-match:relations-label", "precision", "True", prel_lbl_n, prel_lbl_pr],
               ["partial-match:relations-label", "recall", "True", prel_lbl_n, prel_lbl_rl],
               ["full-match:relations-label", "f1", "True", frel_lbl_n, frel_lbl_f1],
               ["full-match:relations-label", "precision", "True", frel_lbl_n, frel_lbl_pr],
               ["full-match:relations-label", "recall", "True", frel_lbl_n, frel_lbl_rl]]

        partial_stats_rpt = classification_report(list(rel_matches_df["gold_tup_rel"].apply(int)),
                                                  list(rel_matches_df["expected_tup_rel_part"].apply(int)),
                                                  zero_division=0, output_dict=True)
        full_stats_rpt = classification_report(list(rel_matches_df["gold_tup_rel"].apply(int)),
                                               list(rel_matches_df["expected_tup_rel_full"].apply(int)),
                                               zero_division=0, output_dict=True)

        label_mapping_rev = {str(k): str(v) for k, v in self.label_mapping_rev.items()}
        for k, v in partial_stats_rpt.items():
            if label_mapping_rev.get(k, None) is not None and k in ["1", "2"]:
                lst.append(["partial-match:relations-label", "f1", label_mapping_rev[k], v["support"],
                            v["f1-score"]])

        for k, v in full_stats_rpt.items():
            if label_mapping_rev.get(k, None) is not None and k in ["1", "2"]:
                lst.append(["full-match:relations-label", "f1", label_mapping_rev[k], v["support"],
                            v["f1-score"]])
        stat_list.extend(lst)
        return stat_list, span_matches_df, rel_matches_df

    def get_span_and_relations(self, output_dct):
        local_arr_gold, global_arr_gold, span_gold, edu_labels_gold, \
            label_probab_lst_gold = None, None, None, None, None
        """ Local Relation Prediction """
        local_head_pred = (torch.sigmoid(output_dct["logits_local_rel_head"].squeeze(0)) >= self.configuration[
            "sigmoid_threshold"]).long().detach().cpu().numpy()
        local_head_pred_app = (torch.sigmoid(output_dct["logits_local_rel_head_app"].squeeze(0)) >= self.configuration[
            "sigmoid_threshold"]).long().detach().cpu().numpy()

        np.fill_diagonal(local_head_pred, 0.0)  # Post processing: Remove self relationships
        np.fill_diagonal(local_head_pred_app, 0.0)  # Post processing: Remove self relationships

        local_edge_pred = output_dct["logits_local_rel_deprel"].squeeze(0).argmax(dim=-1).detach().cpu().numpy()
        local_arr_pred = (local_head_pred * local_edge_pred)

        """ Global Relation Prediction """
        if output_dct.get("logits_global_rel_head", None) is not None:# and output_dct.get("global_rel", None) is not None:
            sigmoid_threshold = self.configuration["sigmoid_threshold"] if \
                self.configuration.get("sigmoid_threshold_global", None) is None else \
                self.configuration["sigmoid_threshold_global"]
            sigmoid_threshold = sigmoid_threshold if sigmoid_threshold is not None else self.configuration["sigmoid_threshold"]
            global_head_pred = (torch.sigmoid(output_dct["logits_global_rel_head"].squeeze(0)) >=
                                sigmoid_threshold).long().detach().cpu().numpy()
            global_edge_pred = output_dct["logits_global_rel_deprel"].squeeze(0).argmax(dim=-1).detach().cpu().numpy()
            global_arr_pred = (global_head_pred * global_edge_pred)
            # np.fill_diagonal(local_arr_pred, 0)  # Post processing: Remove self relationships
            global_arr_pred_probab = torch.softmax(output_dct["logits_global_rel_deprel"].squeeze(0), -1) * \
                                     torch.sigmoid(output_dct["logits_global_rel_head"]).squeeze(0).unsqueeze(-1)

            if self.evaluate and output_dct.get("global_rel", None) is not None:
                global_arr_gold = (output_dct["global_rel"] * output_dct["global_rel_labels"]).squeeze(0).cpu().numpy().astype(int)
            else:
                global_arr_gold = None

        else:
            global_arr_pred, global_arr_gold, global_arr_pred_probab = None, None, None

        """ Segmentation Prediction """
        span_pred = output_dct["logits_segmentation"].argmax(dim=-1).detach().cpu().numpy()

        assert local_head_pred.shape == local_edge_pred.shape
        # assert output_dct["local_rel"].shape == output_dct["local_rel_labels"].shape

        """ Component Classification """
        adu_type_pred_prob, adu_type_pred = F.softmax(output_dct["logits_edu_type"].squeeze(0), -1).max(dim=-1)
        adu_type_pred_prob, adu_type_pred = adu_type_pred_prob.tolist(), adu_type_pred.tolist()

        edu_labels_pred = [self.type_tags_rev[i] for i in adu_type_pred]
        # adu_type_pred_prob = adu_type_pred_prob

        adu_type_pred_prob2, adu_type_pred2 = F.softmax(output_dct["logits_edu_type"].squeeze(0), -1).topk(2)
        adu_type_pred_prob2, adu_type_pred2 = adu_type_pred_prob2[:, 1], adu_type_pred2[:, 1]
        adu_type_pred_prob2, adu_type_pred2 = adu_type_pred_prob2.tolist(), adu_type_pred2.tolist()
        edu_labels_pred2 = [self.type_tags_rev[i] for i in adu_type_pred2]

        if self.evaluate:
            local_arr_gold = (output_dct["local_rel"] * output_dct["local_rel_labels"]).squeeze(0).cpu().numpy().astype(
                int)
            span_gold = output_dct["segment_labels"].detach().cpu().numpy()
            edu_labels_gold = [self.type_tags_rev[i] for i in output_dct["component_labels"].squeeze(0).tolist()]
            assert len(edu_labels_gold) == len(edu_labels_pred) == len(adu_type_pred_prob)
            label_probab_lst_gold = list(zip(edu_labels_gold, [1.0] * len(edu_labels_gold)))
        else:
            assert len(edu_labels_pred) == len(adu_type_pred_prob)

        label_probab_lst_pred = list(zip(edu_labels_pred, adu_type_pred_prob))
        label_probab_lst_pred2 = list(zip(edu_labels_pred2, adu_type_pred_prob2))

        dct = {"local_arr_pred": local_arr_pred, "local_arr_gold": local_arr_gold, "global_arr_pred": global_arr_pred,
               "global_arr_gold": global_arr_gold, "span_pred": span_pred, "span_gold": span_gold,
               "edu_labels_pred": edu_labels_pred, "label_probab_lst_pred": label_probab_lst_pred,
               "edu_labels_pred2": edu_labels_pred2, "label_probab_lst_pred2": label_probab_lst_pred2,
               "edu_labels_gold": edu_labels_gold, "label_probab_lst_gold": label_probab_lst_gold,
               "global_arr_pred_probab": global_arr_pred_probab, "local_head_pred_app": local_head_pred_app}
        dct = {**output_dct, **dct}

        """ local_arr_pred -> Array of PREDICTED local relations of ALL types 
            local_arr_gold -> Array of GOLDEN local relations of ALL types
            global_arr_pred -> Array of PREDICTED global relations of ALL types 
            global_arr_gold -> Array of GOLDEN global relations of ALL types
            span_pred -> Text segmentation prediction
            span_gold -> Text segmentation golden
            edu_labels_pred | label_probab_lst_pred -> Predicted labels of EDUs | along with probabilities
            edu_labels_gold | label_probab_lst_gold -> Golden labels of EDUs | along with probabilities (all 1.0)
        """
        dct = self.get_adu_spans_from_edus(dct)

        """ Map the golden and adu tokens by calculating token overlap. Save the results to overlap_lst.
            overlap_lst contains [uniq_id, golden_adu_id, pred_adu_id, gold_tokens, pred_tokens, 
            gold_adu_label(mc|c|p), pred_adu_label]
            It also contains FP and FN mappings.
            """

        local_from_list_pred, local_to_list_pred = np.where(np.isin(local_arr_pred, [1, 2]))
        local_from_list_pred2 = [dct["local_pred_connected_dict_unrolled"].get(i, -1) for i in local_from_list_pred]
        local_to_list_pred2 = [dct["local_pred_connected_dict_unrolled"].get(i, -1) for i in local_to_list_pred]
        pred_rels = list(set(zip(local_from_list_pred2, local_to_list_pred2)))# [(from, to)]

        pred_rels_map = {}  # (from, to): {rel:rel label, adu_types: (from adu label, to adu label)} adu label:mc|p|c
        for i in pred_rels:
            pred_rels_map[i] = {
                "rel": local_arr_pred[i[0], i[1]]}
        dct["pred_rels_map"] = pred_rels_map

        if self.evaluate:
            dct = self.map_golden_pred_adu_spans(dct)
            local_from_list_gold, local_to_list_gold = np.where(np.isin(local_arr_gold, [1, 2]))
            local_from_list_gold2 = [dct["local_gold_connected_dict_unrolled"].get(i, -1) for i in local_from_list_gold]
            local_to_list_gold2 = [dct["local_gold_connected_dict_unrolled"].get(i, -1) for i in local_to_list_gold]
            gold_rels = list(set(zip(local_from_list_gold2, local_to_list_gold2)))
            gold_rels = [[i, local_arr_gold[i[0], i[1]]] for i in gold_rels]  # gold_rels-> [(from, to), rel label]

            pred_rels_map = {}  # (from, to): {rel:rel label, adu_types: (from adu label, to adu label)} adu label:mc|p|c
            for i in pred_rels:
                pred_rels_map[i] = {
                    "rel": local_arr_pred[i[0], i[1]],
                    "adu_types": (dct["pred_mapping"][i[0]]["type"], dct["pred_mapping"][i[1]]["type"])}

            expected_mapping = []
            # gold (from, to), pred (from, to), pred (from adu token match%, to adu token match%), relationship
            for i in gold_rels:
                f, t = dct["gold2pred"].get(i[0][0], {}), dct["gold2pred"].get(i[0][1], {})
                f_id = f["pred_adu_id"] if f.get("pred_adu_id", None) is not None else None
                t_id = t["pred_adu_id"] if t.get("pred_adu_id", None) is not None else None
                f_pct = f["match_pct"] if f.get("match_pct", None) is not None else 0
                t_pct = t["match_pct"] if t.get("match_pct", None) is not None else 0
                expected_mapping.append([i[0], (f_id, t_id), (f_pct, t_pct), i[1]])

            pairs_done = []
            for expected_lst in expected_mapping:
                gold_mp, expected_mp, expected_mp_pct, expected_rl = expected_lst
                pred_rl = pred_rels_map.get(expected_mp, None)
                pairs_done.append(expected_mp)
                if None in expected_mp or pred_rl is None:
                    tmp = ["fn", "fn", None]
                else:
                    if pred_rl["rel"] == expected_rl:
                        tmp = ["tp", "tp", expected_rl]
                    else:
                        tmp = ["tp", "fn", expected_rl]  # rel exists, rel label is correct, rel label
                expected_lst.extend(tmp)

            for k, i in pred_rels_map.items():
                if k not in pairs_done:
                    expected_mapping.append([(None, None), k, (0, 0), None, "fp", "fp", i["rel"]])
            expected_mapping = [[dct["uniq_id"]] + i for i in expected_mapping]

            dct["expected_mapping"] = expected_mapping  # Relations

        if dct["ctx_input_ids"] is not None:
            global_result, global_txt_lst = self.match_global_adu_edu(dct)
        else:
            global_result, global_txt_lst = [], {}  # []
        dct["global_result"] = global_result
        # dct["global_txt_lst"] = global_txt_lst
        dct["global_txt_dict"] = global_txt_lst

        return dct

    def match_global_adu_edu(self, dct):
        global_from_list_pred, global_to_list_pred = np.where(np.isin(dct["global_arr_pred"], [1, 2]))
        global_pred_rels = list(set(zip(global_from_list_pred, global_to_list_pred)))  # [(from, to)]
        global_pred_rels = [[i, dct["global_arr_pred"][i[0], i[1]]] for i in global_pred_rels]
        df_pred = pd.DataFrame(global_pred_rels, columns=["arg", "rel"])

        results = []
        if self.evaluate and dct["global_rel"] is not None:
            global_from_list_gold, global_to_list_gold = np.where(np.isin(dct["global_arr_gold"], [1, 2]))
            global_gold_rels = list(set(zip(global_from_list_gold, global_to_list_gold)))
            global_gold_rels = [[i, dct["global_arr_gold"][i[0], i[1]]] for i in global_gold_rels]

            df_gold = pd.DataFrame(global_gold_rels, columns=["arg", "rel"])
            df_merged = df_gold.merge(df_pred, how="outer", on="arg")

            for ix, row in df_merged.iterrows():
                tmp = [row["arg"][0], row["arg"][1], row["rel_x"], row["rel_y"]]
                if np.isnan(row["rel_x"]):
                    tmp.extend(["fp", "fp"])
                elif np.isnan(row["rel_y"]):
                    tmp.extend(["fn", "fn"])
                else:
                    if row["rel_x"] == row["rel_y"]:
                        tmp.extend(["tp", "tp"])
                    else:
                        tmp.extend(["tp", "fp"])
                match_pct = 0.0 if dct["gold2pred"].get(row["arg"][1], None) is None else \
                                    dct["gold2pred"][row["arg"][1]]["match_pct"]
                tmp.append(match_pct)

                results.append(tmp)

        """ For finding the text, use connected components & max relationship probability """
        adu_edu_global_txt, global_dct = [], {}
        if len(global_pred_rels) > 0:
            tmp_glob_dict = {}
            max_probs, max_vals = torch.max(dct["global_arr_pred_probab"], -1)
            max_probs, max_vals = max_probs.detach().cpu().numpy(), max_vals.detach().cpu().numpy()
            for i in global_pred_rels:
                ith = i[0][0]
                jth = i[0][1]
                jth_connected = dct["local_pred_connected_dict_unrolled"][jth]
                rel = i[1]
                prob = max_probs[ith, jth]

                existing = tmp_glob_dict.get((ith, jth_connected), None)
                if existing is not None:
                    if prob > existing["prob"]:
                        existing["prob"] = prob
                        existing["rel"] = rel
                else:
                    tmp_glob_dict[(ith, jth_connected)] = {"prob": prob, "rel": rel}

            # for k, v in tmp_glob_dict.items():
            #     global_txt = self.tokenizer.decode(dct["ctx_input_ids"].squeeze(0)[k[0]], skip_special_tokens=True)
            #     local_txt = self.tokenizer.decode(dct["pred_dict_new"][k[1]]["tokens"], skip_special_tokens=False)
            #     local_txt = local_txt.replace("<EDU>", " ").replace("<pad>", "").strip()
            #     rel = self.label_mapping_rev[v["rel"]]
            #     adu_edu_global_txt.append([global_txt, local_txt, rel])

            for ix, (k, v) in enumerate(tmp_glob_dict.items()):
                if v["rel"] in [1, 2]:
                    global_txt = self.tokenizer.decode(dct["ctx_input_ids"].squeeze(0)[k[0]], skip_special_tokens=True)
                    local_txt = self.tokenizer.decode(dct["pred_dict_new"][k[1]]["tokens"], skip_special_tokens=False)
                    local_txt = local_txt.replace("<EDU>", " ").replace("<pad>", "").strip()
                    rel = self.label_mapping_rev[v["rel"]]
                    adu_edu_global_txt.append([global_txt, local_txt, rel])
                    global_dct[ix] = {"global_id": k[0], "global_txt": global_txt,
                                      "local_id": k[1], "local_txt": local_txt, "rel": rel}

            # for i in global_pred_rels:
            #     global_txt = self.tokenizer.decode(dct["ctx_input_ids"][i[0][0]], skip_special_tokens=True)
            #     local_txt = self.tokenizer.decode(dct["pred_dict_new"][i[0][1]]["tokens"], skip_special_tokens=True)
            #     rel = self.label_mapping_rev[i[1]]
            #     adu_edu_global_txt.append([global_txt, local_txt, rel])
        return results, global_dct  # adu_edu_global_txt

    def map_golden_pred_adu_spans(self, variable_dict):
        gold_pred_mapping = self.map_gold_pred(variable_dict["gold_dict_new"], variable_dict["pred_dict_new"])

        variable_dict["gold_dict_new"][-1] = {"tokens": [], "label": -1}
        variable_dict["pred_dict_new"][-1] = {"tokens": [], "label": -1}
        overlap_lst = []
        for i in gold_pred_mapping:
            #   uniq_id, golden_adu_id, pred_adu_id, gold_tokens, pred_tokens, match_pct,
            #                                                               gold_adu_label(mc|c|p), pred_adu_label]
            overlap_lst.append([variable_dict["uniq_id"], i[0], i[1],
                                variable_dict["gold_dict_new"][i[0]]["tokens"],
                                variable_dict["pred_dict_new"][i[1]]["tokens"], i[2],
                                variable_dict["gold_dict_new"][i[0]]["label"],
                                variable_dict["pred_dict_new"][i[1]]["label"]])
        variable_dict["overlap_lst"] = overlap_lst  # Spans

        # gold_mapping, pred_mapping, gold2pred = {}, {}, {}  # {-1: {'pred_adu_id': -1, 'match_pct': 1.0}}  # {-1:-1}
        pred_mapping, gold2pred = {}, {}
        for i in overlap_lst:
            if i[2] != -1:
                pred_mapping[i[2]] = {"tokens": i[4], "type": i[7]}
            if i[1] != -1 and i[2] != -1:
                gold2pred[i[1]] = {"pred_adu_id": i[2], "match_pct": i[-3]}
        variable_dict["gold2pred"] = gold2pred
        variable_dict["pred_mapping"] = pred_mapping
        return variable_dict

    def get_adu_spans_from_edus(self, variable_dict):
        pred_dict = self.get_adu_span_dict(variable_dict["input_ids"], variable_dict["span_pred"])

        """ Find EDUs that should be connected using append relationship to get ADUs """
        local_pred_connected = self.find_connected_segments(variable_dict["local_head_pred_app"])  # self.find_connected_segments(variable_dict["local_arr_pred"])
        # Removes edu ids of non arg tokens from lists of arg tokens
        local_pred_connected = self.fix_connected_component(local_pred_connected, pred_dict)

        """ For each EDU, find the max EDU id from the cluster it belongs to """
        local_pred_connected_dict = {max(i): i for i in local_pred_connected}
        local_pred_connected_dict_unrolled = {i: k for k, v in local_pred_connected_dict.items() for i in v}

        """Update dictionary of ADU spans by combining connected ADU tokens"""
        pred_dict_new = self.consolidate_dict_by_connected_components(local_pred_connected, pred_dict,
                                                                      variable_dict["label_probab_lst_pred"],
                                                                      variable_dict["label_probab_lst_pred2"])
        """ pred_dict_new & gold_dict_new contain ADU level information by consolidating EDUs.
            Structure: {ID: {tokens: [], label: [main_claim|claim|premise|non_arg]}} """

        variable_dict["pred_dict_new"] = pred_dict_new
        variable_dict["local_pred_connected_dict_unrolled"] = local_pred_connected_dict_unrolled

        if self.evaluate:
            gold_dict = self.get_adu_span_dict(variable_dict["input_ids"], variable_dict["span_gold"])
            assert len(gold_dict) == len(pred_dict)

            local_gold_connected = self.find_connected_segments(variable_dict["local_arr_gold"])
            local_gold_connected = self.fix_connected_component(local_gold_connected, gold_dict)

            local_gold_connected_dict = {max(i): i for i in local_gold_connected}
            local_gold_connected_dict_unrolled = {i: k for k, v in local_gold_connected_dict.items() for i in v}

            gold_dict_new = self.consolidate_dict_by_connected_components(local_gold_connected, gold_dict,
                                                                          variable_dict["label_probab_lst_gold"],
                                                                          variable_dict["label_probab_lst_gold"])
            variable_dict["gold_dict_new"] = gold_dict_new
            variable_dict["local_gold_connected_dict_unrolled"] = local_gold_connected_dict_unrolled
        else:
            variable_dict["gold_dict_new"] = None
            variable_dict["local_gold_connected_dict_unrolled"] = None

        return variable_dict

    def get_span_tokens(self, lbl_toks, majority_class):
        span_lst = []
        prev = False
        for ix, i in enumerate(lbl_toks):
            if i[1] == 1:
                span_lst.append(i[0])
                prev = True
            else:
                if ix != 0 and prev and ix + 1 < len(lbl_toks) and lbl_toks[ix + 1][1] == 1 and majority_class == 1:
                    span_lst.append(i[0])
                    prev = True
                else:
                    prev = False
        return span_lst

    def select_predominant(self, zippped_list):
        dct = Counter([i[1] for i in zippped_list])
        if dct[1] >= dct[0]:
            return 1
        else:
            return 0

    def get_adu_span_dict(self, input_ids, span_pred):
        """ Extracts token spans from input_ids, using the span_pred values.
            Also handles irregularities in span labelling by assigning incorrect
            labels to majority class, depending on the affiliation of it's neighbours """
        input_ids = input_ids.view(input_ids.shape[0], -1)#torch.cat(input_ids, -1)
        spans, tmp = [], []
        adu_spans, adu_span_dict = [], {}
        for i in list(zip(input_ids.squeeze(0).tolist(), span_pred.squeeze(0).tolist())):
            if i[0] in self.special_token_idx:
                if len(tmp) > 0:
                    spans.append(tmp)
                    tmp = []
            else:
                tmp.append([i[0], i[1]])
        if len(tmp) > 0:
            spans.append(tmp)

        for lbl_toks in spans:
            majority_class = self.select_predominant(lbl_toks)
            adu_spans.append(self.get_span_tokens(lbl_toks, majority_class))

        for ix, i in enumerate(adu_spans):
            adu_span_dict[ix] = self.special_token_idx + i  # tokenizer.decode(i)

        return adu_span_dict

    def find_connected_segments(self, arr):
        """ Clusters edu_ids whose segments are predicted by be connected by the relationship matrix"""
        edu_list = list(range(arr.shape[0]))
        segments = []
        for edu_id in edu_list:
            if edu_id + 1 in edu_list and arr[edu_id + 1][edu_id] == 1:
                pass
            else:
                vl, idx = 1, edu_id
                tmp = []
                while vl == 1 and idx >= 0:
                    tmp.append(idx)
                    vl = arr[idx][idx - 1]
                    idx -= 1
                segments.append(tmp)
        return segments

    def fix_connected_component(self, comp, dct):
        lst = []
        for i in comp:
            tmp = []
            for j in sorted(i):
                if len(dct[j]) == 0:
                    if len(tmp) > 0:
                        lst.append(tmp)
                    tmp = []
                    lst.append([j])
                else:
                    tmp.append(j)
            if len(tmp) > 0:
                lst.append(tmp)
        assert len(set([j for i in lst for j in i])) == len(set([j for i in comp for j in i]))
        return lst

    def get_best_label(self, lbl_probab):
        df = pd.DataFrame(lbl_probab, columns=["label", "probab"])
        df["freq"] = 1
        df = df.groupby(["label"]).agg({"freq": "sum", "probab": "mean"}).reset_index()
        df = df.sort_values(["freq", "probab"], ascending=False)
        max_freq = df.iloc[0]["freq"]
        max_prob = df.iloc[0]["probab"]
        label_list = list(df[(df["freq"] == max_freq) & (df["probab"] == max_prob)]["label"])

        if "main_claim" in label_list:
            return "main_claim"
        elif "claim" in label_list:
            return "claim"
        else:
            return label_list[0]

    def consolidate_dict_by_connected_components(self, connected_list, dict_old, label_probab_lst,
                                                 label_probab_lst2):
        dict_new = {}
        for i in connected_list:
            tmp, lbl_probab, lbl_probab2 = [], [], []
            for j in sorted(i):
                tmp.extend(dict_old[j])
                lbl_probab.append(label_probab_lst[j])
                lbl_probab2.append(label_probab_lst2[j])
            dict_new[j] = {"tokens": tmp, "label": self.get_best_label(lbl_probab),
                           "second_best_label": self.get_best_label(lbl_probab2)}
        return dict_new

    def find_overlap(self, s1, s2):
        """ Calculates overlap of elements in two lists"""
        s1_toks, s2_toks = set(s1), set(s2)
        if s1 == s2:
            return 1.0

        if len(s1_toks) == 0 and len(s2_toks) == 0:
            return 1.0

        if len(s1_toks) == 0 or len(s2_toks) == 0:
            return 0

        intersect = s1_toks.intersection(s2_toks)
        wrt_s1, wrt_s2 = len(intersect) / len(s1_toks), len(intersect) / len(s2_toks)
        return min([wrt_s1, wrt_s2])

    def map_gold_pred(self, gold_mapping_dict, pred_mapping_dict):
        matched_list, pred_ids_used = [], []
        # For each golden span, find the most matching predicted span.
        for gold_id, golden in gold_mapping_dict.items():
            if golden["label"] != "non_arg":
                max_overlap, max_idx = 0, None
                for pred_id, predicted in pred_mapping_dict.items():
                    # Overlap is performed by matching set of tokens
                    #                 if golden["label"] != "non_arg" and predicted["label"] != "non_arg":
                    overlap = self.find_overlap(golden["tokens"], predicted["tokens"])
                    #                 if len(golden["tokens"]) == 0 and overlap > max_overlap:
                    #                     max_overlap = overlap
                    #                     max_idx = pred_id
                    if overlap >= max_overlap:
                        max_overlap = overlap
                        max_idx = pred_id

                if max_idx is not None:
                    pred_ids_used.append(max_idx)
                matched_list.append([gold_id, max_idx, max_overlap])

        max_map = {}
        for ix, i in enumerate(matched_list):
            if max_map.get(i[1], None) is None:
                max_map[i[1]] = {"max_val": i[-1], "index": ix}
            else:
                if max_map[i[1]]["max_val"] < i[-1]:
                    max_map[i[1]]["max_val"] = i[-1]
                    max_map[i[1]]["index"] = ix

        for ix, i in enumerate(matched_list):
            d = max_map.get(i[1])
            if d["index"] != ix:
                matched_list[ix] = i[:1] + [-1, -1]

        # Additional false positives that did not match at all with the golden spans
        false_positives = [pred_id for pred_id, predicted in pred_mapping_dict.items() if pred_id not in pred_ids_used
                           and predicted["label"] != "non_arg"]
        matched_list.extend([[-1, i, -1] for i in false_positives])

        """ Sanity Check """
        all_gold_identified, all_pred_identified = [], []

        gold_idx = [k for k, v in gold_mapping_dict.items() if v["label"] != "non_arg"]
        pred_idx = [k for k, v in pred_mapping_dict.items() if v["label"] != "non_arg"]

        for i in matched_list:
            if i[0] != -1:
                all_gold_identified.append(i[0])
            if i[1] != -1:
                all_pred_identified.append(i[1])
        assert len(set(gold_idx) - set(all_gold_identified)) == 0 and len(set(pred_idx) - set(all_pred_identified)) == 0
        return matched_list
