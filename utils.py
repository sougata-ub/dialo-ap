""" Utils file containing pre-defined config and commonly used functions """
import torch
from collections import Counter
import random
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import requests
import json

curriculum_prediction_mapping = {
    "C_local_global_component": {
        "predict_segments": False,
        "predict_edu_type": True,
        "predict_local_relations": True,
        "predict_global_relations": True,
        "predict_contextual_relationship": False,
        "batch_size": 8,
        "accumulate": 4
    },
    "C_global_short": {
        "predict_segments": False,
        "predict_edu_type": False,
        "predict_local_relations": False,
        "predict_global_relations": True,
        "predict_contextual_relationship": False,
        "batch_size": 8,
        "accumulate": 4
    },
    "C_global_long": {
        "predict_segments": False,
        "predict_edu_type": False,
        "predict_local_relations": False,
        "predict_global_relations": True,
        "predict_contextual_relationship": False,
        "batch_size": 8,
        "accumulate": 4
    },
    "C_local_global_component_segment": {
        "predict_segments": True,
        "predict_edu_type": True,
        "predict_local_relations": True,
        "predict_global_relations": True,
        "predict_contextual_relationship": False,
        "batch_size": 8,
        "accumulate": 4
    },
    "C_local_component": {
        "predict_segments": False,
        "predict_edu_type": True,
        "predict_local_relations": True,
        "predict_global_relations": False,
        "predict_contextual_relationship": False,
        "batch_size": 6,
        "accumulate": 8
    },
    "C_target_dataset": {
        "predict_segments": True,
        "predict_edu_type": True,
        "predict_local_relations": True,
        "predict_global_relations": True,
        "predict_contextual_relationship": True,
        "batch_size": 1,
        "accumulate": 8
    },
    "pretraining": {
        "predict_segments": False,
        "predict_edu_type": True,
        "predict_local_relations": True,
        "predict_global_relations": False,
        "predict_contextual_relationship": False,
        "batch_size": 64,
        "accumulate": 2
    }
}


def label_match(gold_toks, pred_toks, match_pct, thresh=0.5):
    if match_pct >= thresh:
        if len(gold_toks) == len(pred_toks) == 0:
            return "tn"
        else:
            return "tp"
    else:
        if len(gold_toks) > len(pred_toks):
            return "fn"
        else:
            return "fp"


def df2f1(cnt_dict):
    try:
        return cnt_dict["tp"] / (cnt_dict["tp"] + 0.5 * (cnt_dict["fp"] + cnt_dict["fn"]))
    except Exception as e:
        return 0.0


def df2precision(cnt_dict):
    try:
        return cnt_dict["tp"] / (cnt_dict["tp"] + cnt_dict["fp"])
    except Exception as e:
        return 0.0


def df2recall(cnt_dict):
    try:
        return cnt_dict["tp"] / (cnt_dict["tp"] + cnt_dict["fn"])
    except Exception as e:
        return 0.0


def get_standard_metrics(cnt_dict, get_freq=True):
    f1, pr, rl = df2f1(cnt_dict), df2precision(cnt_dict), df2recall(cnt_dict)
    if get_freq:
        return f1, pr, rl, cnt_dict["tp"]
    else:
        return f1, pr, rl


def get_report_metrics(typ, report, mapping):
    lst = []
    for k, v in report.items():
        if mapping.get(k, None) is not None:
            lst.append([typ, "f1", mapping[k], v["support"], v["f1-score"]])
    lst.append([typ, "f1", "macro avg", report["macro avg"]["support"], report["macro avg"]["f1-score"]])
    return lst


def mark_false_negative(orig_label, pct_tuple, thresh=0.5):
    if pct_tuple[0] >= thresh and pct_tuple[1] >= thresh:
        return orig_label
    else:
        if orig_label == "tp":
            return "fn"
        else:
            return orig_label


def change_rel_label(orig_label, pct_tuple, thresh=0.5):
    if pct_tuple[0] >= thresh and pct_tuple[1] >= thresh:
        return orig_label
    else:
        return -1


# Hosted the NeuralEDUSeg EDU segmenter via a Flask endpoint
def get_segmented_edus(text):
    obj = {"text": text}
    res = requests.post("http://127.0.0.1:9901/generate_edu_segments", json=obj, timeout=2.0)
    return res.json()


def edu_to_text(edu_list, add_special=True):
    if add_special:
        text = ""
        for i in edu_list:
            text += "<EDU>" + i
    else:
        text = " ".join(edu_list)
    return text.strip()
