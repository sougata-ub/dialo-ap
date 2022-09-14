""" Our model architecture """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils.parametrize as parametrize
import copy


class Encoder(nn.Module):
    def __init__(self, transformer, n_layers):
        super().__init__()
        self.enc = transformer
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask=None, token_type=None, position_ids=None):
        hidden = self.enc(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type,
                          position_ids=position_ids)
        token_hidden = hidden["hidden_states"][-self.n_layers:]
        token_hidden = torch.sum(torch.stack(token_hidden), dim=0)  # batch, seq, hidden

        return token_hidden


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ff = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, h):
        return self.dropout(F.relu(self.ff(h)))


class PairwiseBilinear(nn.Module):
    """
    https://github.com/stanfordnlp/stanza/blob/v1.1.1/stanza/models/common/biaffine.py#L5  # noqa
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):#,add_attn: bool = False):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, out_features, in2_features))
#         self.add_attn = add_attn
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor: #, attn: torch.Tensor
        d1, d2, out = self.in1_features, self.in2_features, self.out_features
        n1, n2 = input1.size(1), input2.size(1)
        # (b * n1, d1) @ (d1, out * d2) => (b * n1, out * d2)
        x1W = torch.mm(input1.view(-1, d1), self.weight.view(d1, out * d2))
        # (b, n1 * out, d2) @ (b, d2, n2) => (b, n1 * out, n2)
        x1Wx2 = x1W.view(-1, n1 * out, d2).bmm(input2.transpose(1, 2))
        y = x1Wx2.view(-1, n1, self.out_features, n2).transpose(2, 3)
        if self.bias is not None:
            y.add_(self.bias)
        return y  # (b, n1, n2, out)

    def extra_repr(self) -> str:
        return "in1_features={}, in2_features={}, out_features={}, bias={}".format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )


class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):#, add_attn: bool):
        super().__init__()
        self.bilinear = PairwiseBilinear(in1_features + 1, in2_features + 1, out_features)#, add_attn=add_attn)
        self.bilinear.weight.data.zero_()
        self.bilinear.bias.data.zero_()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor: #, attn: torch.Tensor
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], dim=input1.dim() - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], dim=input2.dim() - 1)
        return self.bilinear(input1, input2)


class Parser(nn.Module):
    def __init__(self, base_transformer, configurations):
        super().__init__()
        self.configurations = configurations
        self.input_encoder = Encoder(base_transformer, configurations["n_layers"])
        # self.ctx_encoder = Encoder(base_transformer, configurations["n_layers"])

        self.self_mha = nn.ModuleList([nn.MultiheadAttention(configurations["in_dim"], configurations["n_heads"],
                                                             batch_first=True, dropout=0.1)
                                       for _ in range(configurations["n_self_attn_layers"])])
        self.self_mha_drop = nn.Dropout(0.1)

        self.ctx_mha = nn.ModuleList([nn.MultiheadAttention(configurations["in_dim"], configurations["n_heads"],
                                                            batch_first=True, dropout=0.1)
                                      for _ in range(configurations["n_self_attn_layers"])])
        self.ctx_mha_drop = nn.Dropout(0.1)

        self.self_ctx_mha = nn.ModuleList([nn.MultiheadAttention(configurations["in_dim"], configurations["n_heads"],
                                                                 batch_first=True, dropout=0.1)
                                           for _ in range(configurations["n_ctx_attn_layers"])])
        self.self_ctx_mha_drop = nn.Dropout(0.1)

        if configurations["predict_contextual_relationship"]:
            self.mlp_src_ctx_lbl = MLP(configurations["in_dim"], configurations["out_dim"])
            self.mlp_trg_ctx_lbl = MLP(configurations["in_dim"], configurations["out_dim"])
            self.biaf_ctx_lbl = Biaffine(configurations["out_dim"], configurations["out_dim"],
                                         configurations["n_context_classes"])

        if configurations["predict_segments"]:
            self.mlp_adu_boundary = MLP(configurations["in_dim"], configurations["out_dim"])
            self.adu_boundary_classifier = nn.Linear(configurations["out_dim"], configurations["n_token_classes"])

        if configurations["predict_edu_type"]:
            self.mlp_edu_type = MLP(configurations["in_dim"], configurations["out_dim"])
            self.type_classifier = nn.Linear(configurations["out_dim"], configurations["n_type_classes"])

        if configurations["predict_local_relations"]:
            self.mlp_src_local_app = MLP(configurations["in_dim"], configurations["out_dim"])
            self.mlp_trg_local_app = MLP(configurations["in_dim"], configurations["out_dim"])
            self.biaf_local_app = Biaffine(configurations["out_dim"], configurations["out_dim"], 1)

            self.mlp_src_local = MLP(configurations["in_dim"], configurations["out_dim"])
            self.mlp_trg_local = MLP(configurations["in_dim"], configurations["out_dim"])
            self.biaf_local = Biaffine(configurations["out_dim"], configurations["out_dim"], 1)

            self.mlp_src_local_lbl = MLP(configurations["in_dim"], configurations["out_dim"])
            self.mlp_trg_local_lbl = MLP(configurations["in_dim"], configurations["out_dim"])
            self.biaf_local_lbl = Biaffine(configurations["out_dim"], configurations["out_dim"],
                                           configurations["n_local_classes"])

        if configurations["predict_global_relations"]:
            self.mlp_src_global = MLP(configurations["in_dim"], configurations["out_dim"])
            self.mlp_trg_global = MLP(configurations["in_dim"], configurations["out_dim"])
            self.biaf_global = Biaffine(configurations["out_dim"], configurations["out_dim"], 1)

            self.mlp_src_global_lbl = MLP(configurations["in_dim"], configurations["out_dim"])
            self.mlp_trg_global_lbl = MLP(configurations["in_dim"], configurations["out_dim"])
            self.biaf_global_lbl = Biaffine(configurations["out_dim"], configurations["out_dim"],
                                            configurations["n_global_classes"])

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                token_type_ids=None,
                edu_idx=None,
                context_input_ids=None,
                context_attention_mask=None,
                context_token_type_ids=None,
                segmentation_labels=None,
                edu_type_labels=None,
                local_rel_head_labels=None,
                local_rel_deprel_labels=None,
                local_rel_app_head_labels=None,
                global_rel_head_labels=None,
                global_rel_deprel_labels=None,
                global_ctx_adu_labels=None
                ):
        """
        :param input_ids: current paragraph input ids sans special tokens, which contain EDU tags => [batch, n_split, seq_len]
        :param position_ids: current paragraph position ids => [batch, n_split, seq_len]
        :param attention_mask: current paragraph attention mask => [batch, n_split, seq_len]
        :param token_type_ids: current paragraph token type denoting the user in dialogical data => [batch, n_split, seq_len] or None
        :param edu_idx: position of the EDUs in input_ids => [n_EDU tags, batch]
        :param context_input_ids: input_ids of the contextual ADUs (not EDUs). Starts with bos & ends with eos token => [batch, n_ctx, seq_len] or None
        :param context_attention_mask: attention_mask of the contextual ADUs => [batch, n_ctx, seq_len] or None
        :param context_token_type_ids: token type of context ADUs denoting the user => [batch, n_ctx, seq_len] or None
        :param segmentation_labels: token B/O tags for input_ids => [batch, n_split * seq_len] or None. Ignore -1 labels
        :param edu_type_labels: EDU NA/MC/C/P tags for EDUs in the current paragraph => [batch, n_EDU tags] or None. Ignore -1 labels
        :param local_rel_head_labels: Binary matrix denoting local relationships between EDUs => [batch, n_EDU, n_EDU tags]. Ignore -1 labels
        :param local_rel_app_head_labels: Binary matrix denoting APPEND between EDUs => [batch, n_EDU, n_EDU tags]. Ignore -1 labels
        :param local_rel_deprel_labels: Labels of the local relationship => [batch, n_EDU, n_EDU tags]. Ignore 0 labels
        :param global_rel_head_labels: Binary matrix denoting global relationships between current EDUs & context ADUs => [batch, n_ADU, n_EDU tags]. Ignore -1 labels
        :param global_rel_deprel_labels: Labels of the global relationship => [batch, n_ADU, n_EDU tags]. Ignore 0 labels
        :param global_ctx_adu_labels: Labels of the relationships between contextual ADUs => [batch, n_ADU ctx, n_ADU ctx]. Ignore 0 labels
        :return: Dictionary containing the logits and losses of all the AM subtasks
        """

        logits_segmentation, logits_edu_type, logits_local_rel_head, logits_local_rel_deprel, logits_global_rel_head, \
            logits_global_rel_deprel, logits_ctx_rel_deprel = None, None, None, None, None, None, None

        loss_segmentation, loss_edu_type, loss_local_rel_head, loss_local_rel_deprel, loss_global_rel_head, \
            loss_global_rel_deprel, loss_ctx_rel_deprel = None, None, None, None, None, None, None

        loss_local_rel_head_app, logits_local_rel_head_app = None, None

        if self.configurations["debug_mode"]:
            print("input_ids: ==>", type(input_ids), "==>", input_ids.shape)
            if attention_mask is not None:
                print("attention_mask: ==>", type(attention_mask), "==>", attention_mask.shape)
            if token_type_ids is not None:
                print("token_type_ids: ==>", type(token_type_ids), "==>", token_type_ids.shape)
            if edu_idx is not None:
                print("edu_idx: ==>", type(edu_idx), "==>", edu_idx.shape)
            if context_input_ids is not None:
                print("context_input_ids: ==>", type(context_input_ids), "==>", context_input_ids.shape)
            if context_attention_mask is not None:
                print("context_attention_mask: ==>", type(context_attention_mask), "==>", context_attention_mask.shape)
            if context_token_type_ids is not None:
                print("context_token_type_ids: ==>", type(context_token_type_ids), "==>", context_token_type_ids.shape)
            if segmentation_labels is not None:
                print("segmentation_labels: ==>", type(segmentation_labels), "==>", segmentation_labels.shape)

            if edu_type_labels is not None:
                print("edu_type_labels: ==>", type(edu_type_labels), "==>", edu_type_labels.shape)
            if local_rel_head_labels is not None:
                print("local_rel_head_labels: ==>", type(local_rel_head_labels), "==>", local_rel_head_labels.shape)
            if local_rel_deprel_labels is not None:
                print("local_rel_deprel_labels: ==>", type(local_rel_deprel_labels), "==>", local_rel_deprel_labels.shape)
            if global_rel_head_labels is not None:
                print("global_rel_head_labels: ==>", type(global_rel_head_labels), "==>", global_rel_head_labels.shape)
            if global_rel_deprel_labels is not None:
                print("global_rel_deprel_labels: ==>", type(global_rel_deprel_labels), "==>", global_rel_deprel_labels.shape)
            if global_ctx_adu_labels is not None:
                print("global_ctx_adu_labels: ==>", type(global_ctx_adu_labels), "==>", global_ctx_adu_labels.shape)

        """ Encode the current input using input_encoder """
        enc = []
        for ix in range(input_ids.shape[1]):
            input_id = input_ids[:, ix, :]
            attn_mask = torch.ones_like(input_id) if attention_mask is None else attention_mask[:, ix, :]
            position_id = None if position_ids is None else position_ids[:, ix, :]
            try:
                enc_tmp = self.input_encoder(input_id, attention_mask=attn_mask,
                                             position_ids=position_id)  # batch, seq_len, in_dim
            except Exception as e:
                print("Exception:", e)
                print("input_id:", input_id.shape)
                raise Exception("Exception!!!")
            enc.append(enc_tmp)
        enc = torch.cat(enc, 1)  # batch, seq (n_split*seq_len), in_dim

        """ Select the EDU representations """
        if edu_idx is not None:
            enc_edu = enc[torch.arange(enc.size(0)), edu_idx].transpose(0, 1)  # batch, n_edu, in_dim
        else:
            enc_edu = enc

        if self.configurations["debug_mode"]:
            print("ENCODED SHAPE ==>", enc.shape)
            print("SELECTED EDU SHAPE ==>", enc_edu.shape, "\n")

        """ MHA with self EDUs """
        if self.configurations["self_mha"]:
            mha_attn_mask = (edu_idx != -1).int().transpose(0, 1).to(enc_edu.device)  # batch, n_edu
            for mha in self.self_mha:
                self_mha_op, _ = mha(query=enc_edu, key=enc_edu, value=enc_edu,
                                     key_padding_mask=mha_attn_mask)  # batch, n_edu, hidden
                enc_edu = enc_edu + self.self_mha_drop(self_mha_op)  # batch, n_edu, hidden

        if self.configurations["debug_mode"]:
            print("DONE WITH CURRENT SELF MHA", "\n")

        """ Encoding context """
        if context_input_ids is not None:
            ctx_enc = []
            for ix in range(context_input_ids.shape[1]):
                context_input_id = context_input_ids[:, ix, :]
                ctx_attn_mask = torch.ones_like(context_input_id) if context_attention_mask is None \
                    else context_attention_mask[:, ix, :]
                # c_enc = self.ctx_encoder(context_input_id, attention_mask=ctx_attn_mask)  # batch, seq, in_dim
                c_enc = self.input_encoder(context_input_id, attention_mask=ctx_attn_mask)[:, :1, :]  # batch, 1, in_dim
                ctx_enc.append(c_enc)  # batch, 1, in_dim
            ctx_enc = torch.cat(ctx_enc, 1)  # batch, n_ctx, in_dim

            """ Context self MHA """
            for mha in self.ctx_mha:
                ctx_mha_op, _ = mha(ctx_enc, ctx_enc, ctx_enc)  # batch, n_ctx, hidden
                ctx_enc = ctx_enc + self.ctx_mha_drop(ctx_mha_op)

            if self.configurations["debug_mode"]:
                print("DONE WITH CONTEXT SELF MHA")
                print("CONTEXT ENC SHAPE ==>", ctx_enc.shape, "\n")

            """ Additional task of learning ctx representation by predicting ctx ADU relationships"""
            if global_ctx_adu_labels is not None and self.configurations["predict_contextual_relationship"]:
                ce_ctx_deprel = nn.CrossEntropyLoss(ignore_index=0)
                hs_src_ctx_lbl = self.mlp_src_ctx_lbl(ctx_enc)  # batch, n_adu, out_dim
                hs_trg_ctx_lbl = self.mlp_trg_ctx_lbl(ctx_enc)  # batch, n_adu, out_dim
                logits_ctx_rel_deprel = self.biaf_ctx_lbl(hs_src_ctx_lbl, hs_trg_ctx_lbl)
                loss_ctx_rel_deprel = ce_ctx_deprel(
                    logits_ctx_rel_deprel.contiguous().view(-1, logits_ctx_rel_deprel.shape[-1]),
                    global_ctx_adu_labels.contiguous().view(-1))

                if self.configurations["debug_mode"]:
                    print("PERFORMED ADDITIONAL CTX RELATIONSHIP PREDICTION TASK", "\n")
        else:
            ctx_enc = None

        """ MHA with current EDUs and context ADUs """
        if self.configurations["context_attention"] and ctx_enc is not None:
            # mha_attn_mask = (edu_idx != -1).int().transpose(0, 1).to(enc_edu.device)  # batch, n_edu
            for mha in self.self_ctx_mha:
                self_ctx_mha_op, _ = mha(enc_edu, ctx_enc, ctx_enc)#,key_padding_mask=mha_attn_mask)  # batch, n_edu, hidden
                enc_edu = enc_edu + self.self_ctx_mha_drop(self_ctx_mha_op)

            if self.configurations["debug_mode"]:
                print("PERFORMED MHA BETWEEN CURRENT EDUs AND CONTEXT ADUs", "\n")

        """ Perform text segmentation """
        if segmentation_labels is not None and self.configurations["predict_segments"]:
            ce_seg = nn.CrossEntropyLoss(ignore_index=-1)
            logits_segmentation = self.adu_boundary_classifier(self.mlp_adu_boundary(enc))  # batch, seq, tok_type
            pred = logits_segmentation.contiguous().view(-1, logits_segmentation.shape[-1])
            tgt = segmentation_labels.contiguous().view(-1)
            if self.configurations["debug_mode"]:
                print("PERFORMING SEGMENTATION")
                print("PRED ==>", pred.shape, "TARGET ==>", tgt.shape, "\n")
            loss_segmentation = ce_seg(pred, tgt)

        """ Perform EDU Type Classification """
        if edu_type_labels is not None and self.configurations["predict_edu_type"]:
            ce_typ = nn.CrossEntropyLoss(ignore_index=-1)
            logits_edu_type = self.type_classifier(self.mlp_edu_type(enc_edu))  # batch, n_edu, adu_type
            pred = logits_edu_type.contiguous().view(-1, logits_edu_type.shape[-1])
            tgt = edu_type_labels.contiguous().view(-1)
            if self.configurations["debug_mode"]:
                print("PERFORMING COMPONENT CLASSIFICATION")
                print("PRED ==>", pred.shape, "TARGET ==>", tgt.shape, "\n")
            loss_edu_type = ce_typ(pred, tgt)

        """ Perform Local relationship detection """
        if local_rel_head_labels is not None and self.configurations["predict_local_relations"]:
            pos_weight = torch.ones_like(local_rel_head_labels).view(-1) * self.configurations["positive_class_weight"]#torch.tensor([1.0, 5.0]).to(local_rel_head_labels.device)#torch.ones_like(local_rel_head_labels).view(-1) * 5.0
            bce_local_head = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            bce_local_head_app = nn.BCEWithLogitsLoss(reduction="none")
            # bce_local_head = nn.CrossEntropyLoss(weight=pos_weight, ignore_index=-1)
            # bce_local_head_app = nn.CrossEntropyLoss(ignore_index=-1)

            hs_src_local = self.mlp_src_local(enc_edu)  # batch, n_edu, out_dim
            hs_trg_local = self.mlp_trg_local(enc_edu)  # batch, n_edu, out_dim

            hs_src_local_app = self.mlp_src_local_app(enc_edu)  # batch, n_edu, out_dim
            hs_trg_local_app = self.mlp_trg_local_app(enc_edu)  # batch, n_edu, out_dim

            logits_local_rel_head = self.biaf_local(hs_src_local, hs_trg_local).squeeze_(3)  # (batch, n_edu, n_edu)
            logits_local_rel_head_app = self.biaf_local_app(hs_src_local_app, hs_trg_local_app).squeeze_(3)  # (batch, n_edu, n_edu)

            pred = logits_local_rel_head.contiguous().view(-1)#, logits_local_rel_head.shape[-1])
            tgt = local_rel_head_labels.contiguous().view(-1)
            tgt_mask = tgt != -1

            pred_app = logits_local_rel_head_app.contiguous().view(-1)#, logits_local_rel_head_app.shape[-1])
            tgt_app = local_rel_app_head_labels.contiguous().view(-1)
            tgt_app_mask = tgt_app != -1

            if self.configurations["debug_mode"]:
                print("PERFORMING LOCAL REL DETECTION")
                print("PRED ==>", pred.shape, "TARGET ==>", tgt.shape, "\n")
                print("PRED ==>", pred, "TARGET ==>", tgt, "\n")

            loss_local_rel_head = bce_local_head(pred, tgt)
            loss_local_rel_head = torch.masked_select(loss_local_rel_head, tgt_mask).mean()

            loss_local_rel_head_app = bce_local_head_app(pred_app, tgt_app)
            loss_local_rel_head_app = torch.masked_select(loss_local_rel_head_app, tgt_app_mask).mean()

        """ Perform Local relationship labelling """
        if local_rel_deprel_labels is not None and self.configurations["predict_local_relations"]:
            ce_local_deprel = nn.CrossEntropyLoss(ignore_index=0)
            hs_src_local_lbl = self.mlp_src_local_lbl(enc_edu)  # batch, n_edu, out_dim
            hs_trg_local_lbl = self.mlp_trg_local_lbl(enc_edu)  # batch, n_edu, out_dim
            logits_local_rel_deprel = self.biaf_local_lbl(hs_src_local_lbl, hs_trg_local_lbl) # (batch, n_edu, n_edu, n_deprel_classes)

            pred = logits_local_rel_deprel.contiguous().view(-1, logits_local_rel_deprel.shape[-1])
            tgt = local_rel_deprel_labels.contiguous().view(-1)

            if self.configurations["debug_mode"]:
                print("PERFORMING LOCAL REL LABELLING")
                print("PRED ==>", pred.shape, "TARGET ==>", tgt.shape, "\n")

            loss_local_rel_deprel = ce_local_deprel(pred, tgt)

        """ Perform Global relationship detection """
        if global_rel_head_labels is not None and self.configurations["predict_global_relations"] and ctx_enc is not None:
            pos_weight = torch.ones_like(global_rel_head_labels).view(-1) * self.configurations["global_positive_class_weight"]
            bce_global_head = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)  # nn.CrossEntropyLoss(ignore_index=-1)  # nn.BCEWithLogitsLoss()
            hs_src_global = self.mlp_src_global(ctx_enc)  # batch, n_ctx_adu, out_dim
            hs_trg_global = self.mlp_trg_global(enc_edu)  # batch, n_edu, out_dim
            logits_global_rel_head = self.biaf_global(hs_src_global, hs_trg_global).squeeze_(3)  # (batch, n_ctx_adu, n_edu)
            # print(logits_global_rel_head.shape, global_rel_head_labels.shape, global_rel_deprel_labels.shape)
            pred = logits_global_rel_head.contiguous().view(-1) #, logits_global_rel_head.shape[-1])
            tgt = global_rel_head_labels.contiguous().view(-1)
            tgt_mask = tgt != -1
            if self.configurations["debug_mode"]:
                print("PERFORMED GLOBAL REL DETECTION")
                print("PRED ==>", pred.shape, "TARGET ==>", tgt.shape, "\n")

            loss_global_rel_head = bce_global_head(pred, tgt)
            loss_global_rel_head = torch.masked_select(loss_global_rel_head, tgt_mask).mean()

        """ Perform Global relationship labelling """
        if global_rel_deprel_labels is not None and self.configurations["predict_global_relations"] and ctx_enc is not None:
            ce_global_deprel = nn.CrossEntropyLoss(ignore_index=0)
            hs_src_global_lbl = self.mlp_src_global_lbl(ctx_enc)
            hs_trg_global_lbl = self.mlp_trg_global_lbl(enc_edu)  # batch, n_edu, out_dim
            logits_global_rel_deprel = self.biaf_global_lbl(hs_src_global_lbl, hs_trg_global_lbl)  # (batch, n_ctx_adu, n_edu, n_deprel_classes)
            pred = logits_global_rel_deprel.contiguous().view(-1, logits_global_rel_deprel.shape[-1])
            tgt = global_rel_deprel_labels.contiguous().view(-1)
            if self.configurations["debug_mode"]:
                print("PERFORMING GLOBAL REL LABELLING")
                print("PRED ==>", pred.shape, "TARGET ==>", tgt.shape, "\n")

            loss_global_rel_deprel = ce_global_deprel(pred, tgt)

        dct = {"logits_segmentation": logits_segmentation, "logits_edu_type": logits_edu_type,
               "logits_local_rel_head": logits_local_rel_head, "logits_local_rel_deprel": logits_local_rel_deprel,
               "logits_global_rel_head": logits_global_rel_head, "logits_global_rel_deprel": logits_global_rel_deprel,
               "logits_ctx_rel_deprel": logits_ctx_rel_deprel, "loss_segmentation": loss_segmentation,
               "logits_local_rel_head_app": logits_local_rel_head_app,
               "loss_edu_type": loss_edu_type, "loss_local_rel_head": loss_local_rel_head,
               "loss_local_rel_deprel": loss_local_rel_deprel, "loss_global_rel_head": loss_global_rel_head,
               "loss_global_rel_deprel": loss_global_rel_deprel, "loss_ctx_rel_deprel": loss_ctx_rel_deprel,
               "loss_local_rel_head_app": loss_local_rel_head_app}

        if self.configurations["debug_mode"]:
            print("RETURNING VALUES ==>", dct, "\n")
        return dct

    def predict(self, input_ids, position_ids=None, attention_mask=None, token_type_ids=None, edu_idx=None,
                context_input_ids=None, context_attention_mask=None, context_token_type_ids=None):

        logits_segmentation, logits_edu_type, logits_local_rel_head, logits_local_rel_deprel, logits_global_rel_head, \
            logits_global_rel_deprel, logits_ctx_rel_deprel, \
                logits_local_rel_head_app = None, None, None, None, None, None, None, None

        if self.configurations["debug_mode"]:
            print("input_ids: ==>", type(input_ids), "==>", input_ids)
            print("attention_mask: ==>", type(attention_mask), "==>", attention_mask)
            print("token_type_ids: ==>", type(token_type_ids), "==>", token_type_ids)
            print("edu_idx: ==>", type(edu_idx), "==>", edu_idx)

            print("context_input_ids: ==>", type(context_input_ids), "==>", context_input_ids)
            print("context_attention_mask: ==>", type(context_attention_mask), "==>", context_attention_mask)
            print("context_token_type_ids: ==>", type(context_token_type_ids), "==>", context_token_type_ids)

        """ Encode the current input using input_encoder """
        enc = []
        for ix in range(input_ids.shape[1]):
            input_id = input_ids[:, ix, :]
            attn_mask = torch.ones_like(input_id) if attention_mask is None else attention_mask[:, ix, :]
            position_id = None if position_ids is None else position_ids[:, ix, :]
            enc_tmp = self.input_encoder(input_id, attention_mask=attn_mask,
                                         position_ids=position_id)  # batch, seq_len, in_dim
            enc.append(enc_tmp)
        enc = torch.cat(enc, 1)  # batch, seq (n_split*seq_len), in_dim

        """ Select the EDU representations """
        if edu_idx is not None:
            enc_edu = enc[torch.arange(enc.size(0)), edu_idx].transpose(0, 1)  # batch, n_edu, in_dim
        else:
            enc_edu = enc

        if self.configurations["debug_mode"]:
            print("ENCODED SHAPE ==>", enc.shape)
            print("SELECTED EDU SHAPE ==>", enc_edu.shape, "\n")

        """ MHA with self EDUs """
        if self.configurations["self_mha"]:
            mha_attn_mask = (edu_idx != -1).int().transpose(0, 1).to(enc_edu.device)  # batch, n_edu
            for mha in self.self_mha:
                self_mha_op, _ = mha(query=enc_edu, key=enc_edu, value=enc_edu,
                                     key_padding_mask=mha_attn_mask)  # batch, n_edu, hidden
                enc_edu = enc_edu + self.self_mha_drop(self_mha_op)  # batch, n_edu, hidden

        if self.configurations["debug_mode"]:
            print("DONE WITH CURRENT SELF MHA", "\n")

        """ Encoding context """
        if context_input_ids is not None:
            ctx_enc = []
            for ix in range(context_input_ids.shape[1]):
                context_input_id = context_input_ids[:, ix, :]
                ctx_attn_mask = torch.ones_like(context_input_id) if context_attention_mask is None \
                    else context_attention_mask[:, ix, :]
                c_enc = self.input_encoder(context_input_id, attention_mask=ctx_attn_mask)  # batch, seq, in_dim
                ctx_enc.append(c_enc[:, :1, :])  # batch, 1, in_dim
            ctx_enc = torch.cat(ctx_enc, 1)  # batch, n_ctx, in_dim

            """ Context self MHA """
            for mha in self.ctx_mha:
                ctx_mha_op, _ = mha(ctx_enc, ctx_enc, ctx_enc)  # batch, n_ctx, hidden
                ctx_enc = ctx_enc + self.ctx_mha_drop(ctx_mha_op)

            if self.configurations["debug_mode"]:
                print("DONE WITH CONTEXT SELF MHA")
                print("CONTEXT ENC SHAPE ==>", ctx_enc.shape, "\n")

            """ Additional task of learning ctx representation by predicting ctx ADU relationships"""
            if self.configurations["predict_contextual_relationship"]:
                hs_src_ctx_lbl = self.mlp_src_ctx_lbl(ctx_enc)  # batch, n_adu, out_dim
                hs_trg_ctx_lbl = self.mlp_trg_ctx_lbl(ctx_enc)  # batch, n_adu, out_dim
                logits_ctx_rel_deprel = self.biaf_ctx_lbl(hs_src_ctx_lbl, hs_trg_ctx_lbl)

                if self.configurations["debug_mode"]:
                    print("PERFORMED ADDITIONAL CTX RELATIONSHIP PREDICTION TASK", "\n")
        else:
            ctx_enc = None

        """ MHA with current EDUs and context ADUs """
        if self.configurations["context_attention"] and ctx_enc is not None:
            # mha_attn_mask = (edu_idx != -1).int().transpose(0, 1).to(enc_edu.device)  # batch, n_edu
            for mha in self.self_ctx_mha:
                self_ctx_mha_op, _ = mha(enc_edu, ctx_enc, ctx_enc)#,key_padding_mask=mha_attn_mask)  # batch, n_edu, hidden
                enc_edu = enc_edu + self.self_ctx_mha_drop(self_ctx_mha_op)

            if self.configurations["debug_mode"]:
                print("PERFORMED MHA BETWEEN CURRENT EDUs AND CONTEXT ADUs", "\n")

        """ Perform text segmentation """
        if self.configurations["predict_segments"]:
            logits_segmentation = self.adu_boundary_classifier(self.mlp_adu_boundary(enc))  # batch, seq, tok_type
            pred = logits_segmentation.contiguous().view(-1, logits_segmentation.shape[-1])

            if self.configurations["debug_mode"]:
                print("PERFORMING SEGMENTATION")
                print("PRED ==>", pred.shape, "\n")

        """ Perform EDU Type Classification """
        if self.configurations["predict_edu_type"]:
            logits_edu_type = self.type_classifier(self.mlp_edu_type(enc_edu))  # batch, n_edu, adu_type
            pred = logits_edu_type.contiguous().view(-1, logits_edu_type.shape[-1])
            if self.configurations["debug_mode"]:
                print("PERFORMING COMPONENT CLASSIFICATION")
                print("PRED ==>", pred.shape, "\n")

        """ Perform Local relationship detection """
        if self.configurations["predict_local_relations"]:
            hs_src_local = self.mlp_src_local(enc_edu)  # batch, n_edu, out_dim
            hs_trg_local = self.mlp_trg_local(enc_edu)  # batch, n_edu, out_dim

            hs_src_local_app = self.mlp_src_local_app(enc_edu)  # batch, n_edu, out_dim
            hs_trg_local_app = self.mlp_trg_local_app(enc_edu)  # batch, n_edu, out_dim

            logits_local_rel_head = self.biaf_local(hs_src_local, hs_trg_local).squeeze_(3)  # (batch, n_edu, n_edu)
            logits_local_rel_head_app = self.biaf_local_app(hs_src_local_app, hs_trg_local_app).squeeze_(3)  # (batch, n_edu, n_edu)

            if self.configurations["debug_mode"]:
                print("PERFORMING LOCAL REL DETECTION")
                print("PRED REL ==>", logits_local_rel_head.shape, "\n")
                print("PRED REL APP ==>", logits_local_rel_head_app, "\n")

        """ Perform Local relationship labelling """
        if self.configurations["predict_local_relations"]:
            hs_src_local_lbl = self.mlp_src_local_lbl(enc_edu)  # batch, n_edu, out_dim
            hs_trg_local_lbl = self.mlp_trg_local_lbl(enc_edu)  # batch, n_edu, out_dim
            logits_local_rel_deprel = self.biaf_local_lbl(hs_src_local_lbl, hs_trg_local_lbl) # (batch, n_edu, n_edu, n_deprel_classes)

            if self.configurations["debug_mode"]:
                print("PERFORMING LOCAL REL LABELLING")
                print("PRED ==>", logits_local_rel_deprel.shape, "\n")

        """ Perform Global relationship detection """
        if self.configurations["predict_global_relations"] and ctx_enc is not None:
            hs_src_global = self.mlp_src_global(ctx_enc)  # batch, n_ctx_adu, out_dim
            hs_trg_global = self.mlp_trg_global(enc_edu)  # batch, n_edu, out_dim
            logits_global_rel_head = self.biaf_global(hs_src_global, hs_trg_global).squeeze_(3)  # (batch, n_ctx_adu, n_edu)

            if self.configurations["debug_mode"]:
                print("PERFORMED GLOBAL REL DETECTION")
                print("PRED ==>", logits_global_rel_head.shape, "\n")

        """ Perform Global relationship labelling """
        if self.configurations["predict_global_relations"] and ctx_enc is not None:
            hs_src_global_lbl = self.mlp_src_global_lbl(ctx_enc)
            hs_trg_global_lbl = self.mlp_trg_global_lbl(enc_edu)  # batch, n_edu, out_dim
            logits_global_rel_deprel = self.biaf_global_lbl(hs_src_global_lbl, hs_trg_global_lbl)  # (batch, n_ctx_adu, n_edu, n_deprel_classes)

            if self.configurations["debug_mode"]:
                print("PERFORMING GLOBAL REL LABELLING")
                print("PRED ==>", logits_global_rel_deprel.shape, "\n")

        dct = {"logits_segmentation": logits_segmentation, "logits_edu_type": logits_edu_type,
               "logits_local_rel_head": logits_local_rel_head, "logits_local_rel_deprel": logits_local_rel_deprel,
               "logits_global_rel_head": logits_global_rel_head, "logits_global_rel_deprel": logits_global_rel_deprel,
               "logits_ctx_rel_deprel": logits_ctx_rel_deprel, "logits_local_rel_head_app": logits_local_rel_head_app}

        if self.configurations["debug_mode"]:
            print("RETURNING VALUES ==>", dct, "\n")
        return dct
