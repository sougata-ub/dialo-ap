""" E2E Parser built using the recreated Ampersand classifiers """

from nltk.tokenize import sent_tokenize
from transformers import BertForPreTraining, BertTokenizerFast
import torch
import torch.nn as nn
from ampersand_baseline_trainer import Model
from torch.utils.data import DataLoader, SequentialSampler
from collections import OrderedDict
import os
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class AmpersandArgParser:
    def __init__(self, device, component_pretrained, intra_pretrained, inter_pretrained, batch_size=64, debug=True):
        self.device = device
        self.batch_size = batch_size
        self.debug = debug
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.component_predictor, self.intra_predictor, \
            self.inter_predictor = self.load_model(component_pretrained, intra_pretrained, inter_pretrained)
        self.component_map = {0: "non_arg", 1: "claim", 2: "premise"}

    def load_model(self, component_pretrained, intra_pretrained, inter_pretrained):
        base_model = BertForPreTraining.from_pretrained("bert-base-uncased", output_hidden_states=True)
        component_model = Model(copy.deepcopy(base_model), "component").to(self.device)
        intra_model = Model(copy.deepcopy(base_model), "intra").to(self.device)
        inter_model = Model(copy.deepcopy(base_model), "inter").to(self.device)

        state_dict = torch.load(component_pretrained)
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        component_model.load_state_dict(state_dict)

        state_dict = torch.load(intra_pretrained)
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        intra_model.load_state_dict(state_dict)

        state_dict = torch.load(inter_pretrained)
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        inter_model.load_state_dict(state_dict)

        return component_model, intra_model, inter_model

    def data_process(self, data_dict):
        data_lst = []
        for ix, i in enumerate(data_dict["input_ids"]):
            data_lst.append([i, data_dict["token_type_ids"][ix], data_dict["attention_mask"][ix]])
        return data_lst

    def get_dataloader(self, data_dict):
        dataset_processed = self.data_process(data_dict)
        dataloader = DataLoader(dataset_processed, batch_size=self.batch_size,
                                sampler=SequentialSampler(dataset_processed), num_workers=2)
        return dataloader

    def process_text(self, text_list, pair=False):
        if pair:
            sent_a, sent_b = text_list
            inputs = self.tokenizer(sent_a, sent_b, max_length=200, truncation=True,
                                    return_tensors="pt", padding=True)
        else:
            inputs = self.tokenizer(text_list, max_length=200, truncation=True,
                                    return_tensors="pt", padding=True)
        dct = {"input_ids": inputs.input_ids, "token_type_ids": inputs.token_type_ids,
               "attention_mask": inputs.attention_mask}
        return dct

    def predict(self, text_list, type):
        if type in ["inter", "intra"]:
            sent_a, sent_b = [], []
            for i in text_list:
                sent_a.append(i[0])
                sent_b.append(i[1])
            data_dict = self.process_text((sent_a, sent_b), pair=True)
        else:
            data_dict = self.process_text(text_list, pair=False)

        dataloader = self.get_dataloader(data_dict)

        pred, pred_logits = [], []
        for ix, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, type_ids, attention_masks = batch
            with torch.no_grad():
                if type == "component":
                    outputs = self.component_predictor(input_ids, token_type_ids=type_ids, attention_mask=attention_masks)
                elif type == "intra":
                    outputs = self.intra_predictor(input_ids, token_type_ids=type_ids, attention_mask=attention_masks)
                else:
                    outputs = self.inter_predictor(input_ids, token_type_ids=type_ids, attention_mask=attention_masks)
            pred.extend(outputs["prediction_logits"].argmax(dim=-1).detach().view(-1).tolist())
            pred_logits.extend(torch.softmax(outputs["prediction_logits"], -1).detach().tolist())
        return pred, pred_logits

    def run(self, conversation):
        turns = [i.strip() for i in conversation.split("\t")]
        context, context_idx = [], []
        all_component_dict, all_intra_dict, all_inter_dict = OrderedDict(), OrderedDict(), OrderedDict()
        all_intra_dict_lgt, all_inter_dict_lgt = OrderedDict(), OrderedDict()

        for t_ix, turn in enumerate(turns):
            turn_id = "turn:"+str(t_ix)

            sentences = [i.strip() for i in sent_tokenize(turn)]
            """ Predict C/P/NA """
            sentence_components, sentence_components_logits = self.predict(sentences, type="component")
            sentence_components = [self.component_map[i] for i in sentence_components]

            claim_sentences, premise_sentences = [], []
            claim_sentences_idx, premise_sentences_idx = [], []
            for s_ix, i in enumerate(sentence_components):
                comp_idx = turn_id + ":component:" + str(s_ix)
                all_component_dict[comp_idx] = {"text": sentences[s_ix], "type": i}
                if self.debug:
                    all_component_dict[comp_idx]["logits"] = sentence_components_logits[s_ix]

                if i == "claim":
                    claim_sentences.append(i)
                    claim_sentences_idx.append(comp_idx)
                elif i == "premise":
                    premise_sentences.append(i)
                    premise_sentences_idx.append(comp_idx)

            """ Predict Intra """
            if len(claim_sentences) > 0 and len(premise_sentences) > 0:
                claim_premise_pairs, claim_premise_pairs_idx = [], []
                for ix1, claim in enumerate(claim_sentences):
                    for ix2, premise in enumerate(premise_sentences):
                        claim_premise_pairs.append([claim, premise])
                        claim_premise_pairs_idx.append([claim_sentences_idx[ix1], premise_sentences_idx[ix2]])

                intra_predictions, intra_predictions_logits = self.predict(claim_premise_pairs, type="intra")
                for idx, i in enumerate(intra_predictions):
                    if i == 1:
                        all_intra_dict[len(all_intra_dict) + 1] = {"target_component": all_component_dict[claim_premise_pairs_idx[idx][0]],
                                                                   "source_component": all_component_dict[claim_premise_pairs_idx[idx][1]]}
                    if self.debug:
                        all_intra_dict_lgt[len(all_intra_dict_lgt) + 1] = {"target_component": all_component_dict[claim_premise_pairs_idx[idx][0]],
                                                                           "source_component": all_component_dict[claim_premise_pairs_idx[idx][1]],
                                                                           "logit": intra_predictions_logits[idx]}

            """ Predict Inter """
            if len(context) > 0 and len(claim_sentences) > 0:
                context_claim_pairs, context_claim_pairs_idx = [], []
                for ix1, con in enumerate(context):
                    for ix2, claim in enumerate(claim_sentences):
                        context_claim_pairs.append([con, claim])
                        context_claim_pairs_idx.append([context_idx[ix1], claim_sentences_idx[ix2]])

                inter_predictions, inter_predictions_logits = self.predict(context_claim_pairs, type="inter")
                for idx, i in enumerate(inter_predictions):
                    if i == 1:
                        all_inter_dict[len(all_inter_dict) + 1] = {"target_component": all_component_dict[context_claim_pairs_idx[idx][0]],
                                                                   "source_component": all_component_dict[context_claim_pairs_idx[idx][1]]}

                    if self.debug:
                        all_inter_dict_lgt[len(all_inter_dict_lgt) + 1] = {"target_component": all_component_dict[context_claim_pairs_idx[idx][0]],
                                                                           "source_component": all_component_dict[context_claim_pairs_idx[idx][1]],
                                                                           "logit": inter_predictions_logits[idx]}

            """ Add arg components to context"""
            if len(claim_sentences + premise_sentences) > 0:
                context.extend(claim_sentences + premise_sentences)
                context_idx.extend(claim_sentences_idx + premise_sentences_idx)

        op_dct = {"all_components": all_component_dict, "all_intra_relationships": all_intra_dict,
                  "all_inter_relationships": all_inter_dict}
        if self.debug:
            op_dct["all_intra_relationships_logits"] = all_intra_dict_lgt
            op_dct["all_inter_relationships_logits"] = all_inter_dict_lgt

        return op_dct

    def run_v2(self, turns):
        context, context_idx, context_user = [], [], []
        all_component_dict, all_intra_dict, all_inter_dict = OrderedDict(), OrderedDict(), OrderedDict()
        all_intra_dict_lgt, all_inter_dict_lgt = OrderedDict(), OrderedDict()

        for t_ix, text in enumerate(turns):
            user_id, turn = text
            turn_id = "turn:"+str(t_ix)+":user"+str(user_id)

            sentences = [i.strip() for i in sent_tokenize(turn)]
            """ Predict C/P/NA """
            sentence_components, sentence_components_logits = self.predict(sentences, type="component")
            sentence_components = [self.component_map[i] for i in sentence_components]

            claim_sentences, premise_sentences = [], []
            claim_sentences_idx, premise_sentences_idx = [], []
            for s_ix, i in enumerate(sentence_components):
                comp_idx = turn_id + ":component:" + str(s_ix)
                all_component_dict[comp_idx] = {"text": sentences[s_ix], "type": i}
                if self.debug:
                    all_component_dict[comp_idx]["logits"] = sentence_components_logits[s_ix]

                if i == "claim":
                    claim_sentences.append(i)
                    claim_sentences_idx.append(comp_idx)
                elif i == "premise":
                    premise_sentences.append(i)
                    premise_sentences_idx.append(comp_idx)

            """ Predict Intra """
            if len(claim_sentences) > 0 and len(premise_sentences) > 0:
                claim_premise_pairs, claim_premise_pairs_idx = [], []
                for ix1, claim in enumerate(claim_sentences):
                    for ix2, premise in enumerate(premise_sentences):
                        claim_premise_pairs.append([claim, premise])
                        claim_premise_pairs_idx.append([claim_sentences_idx[ix1], premise_sentences_idx[ix2]])

                intra_predictions, intra_predictions_logits = self.predict(claim_premise_pairs, type="intra")
                for idx, i in enumerate(intra_predictions):
                    if i == 1:
                        all_intra_dict[len(all_intra_dict) + 1] = {"target_component": all_component_dict[claim_premise_pairs_idx[idx][0]],
                                                                   "source_component": all_component_dict[claim_premise_pairs_idx[idx][1]]}
                    if self.debug:
                        all_intra_dict_lgt[len(all_intra_dict_lgt) + 1] = {"target_component": all_component_dict[claim_premise_pairs_idx[idx][0]],
                                                                           "source_component": all_component_dict[claim_premise_pairs_idx[idx][1]],
                                                                           "logit": intra_predictions_logits[idx]}

            """ Predict Inter """
            if len(context) > 0 and len(claim_sentences) > 0:
                context_claim_pairs, context_claim_pairs_idx = [], []
                for ix1, con in enumerate(context):
                    if str(context_user[ix1]) != str(user_id):
                        for ix2, claim in enumerate(claim_sentences):
                            context_claim_pairs.append([con, claim])
                            context_claim_pairs_idx.append([context_idx[ix1], claim_sentences_idx[ix2]])

                if len(context_claim_pairs) > 0:
                    inter_predictions, inter_predictions_logits = self.predict(context_claim_pairs, type="inter")
                    for idx, i in enumerate(inter_predictions):
                        if i == 1:
                            all_inter_dict[len(all_inter_dict) + 1] = {"target_component": all_component_dict[context_claim_pairs_idx[idx][0]],
                                                                       "source_component": all_component_dict[context_claim_pairs_idx[idx][1]]}

                        if self.debug:
                            all_inter_dict_lgt[len(all_inter_dict_lgt) + 1] = {"target_component": all_component_dict[context_claim_pairs_idx[idx][0]],
                                                                               "source_component": all_component_dict[context_claim_pairs_idx[idx][1]],
                                                                               "logit": inter_predictions_logits[idx]}

            """ Add arg components to context"""
            arg_sents = claim_sentences + premise_sentences
            if len(arg_sents) > 0:
                context.extend(arg_sents)
                context_idx.extend(claim_sentences_idx + premise_sentences_idx)
                context_user.extend([user_id] * len(arg_sents))

        op_dct = {"all_components": all_component_dict, "all_intra_relationships": all_intra_dict,
                  "all_inter_relationships": all_inter_dict}
        if self.debug:
            op_dct["all_intra_relationships_logits"] = all_intra_dict_lgt
            op_dct["all_inter_relationships_logits"] = all_inter_dict_lgt

        return op_dct
