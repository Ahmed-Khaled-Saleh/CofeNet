import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from infer.model.base import ExpModelBase
from infer.model.torch_utils import WordBert, EnhancedCell
# from infer.model.torch_utils import sequence_mask


# class ModelBert(ExpModelBase):
#     def __init__(self):
#         super(ModelBert, self).__init__()
#         self.tag_size = 7 #config['tag_size']

#         self.layer_bert = WordBert({
#                                     "architectures": [
#                                         "BertForMaskedLM"
#                                     ],
#                                     "attention_probs_dropout_prob": 0.1,
#                                     "finetuning_task": None,
#                                     "hidden_act": "gelu",
#                                     "hidden_dropout_prob": 0.1,
#                                     "hidden_size": 768,
#                                     "initializer_range": 0.02,
#                                     "intermediate_size": 3072,
#                                     "layer_norm_eps": 1e-12,
#                                     "max_position_embeddings": 512,
#                                     "model_type": "bert",
#                                     "num_attention_heads": 12,
#                                     "num_hidden_layers": 12,
#                                     "num_labels": 2,
#                                     "output_attentions": False,
#                                     "output_hidden_states": False,
#                                     "pad_token_id": 0,
#                                     "pruned_heads": {},
#                                     "torchscript": False,
#                                     "type_vocab_size": 2,
#                                     "vocab_size": 30522
#                                     })

#         self.layer_output = nn.Linear(self.layer_bert.bert_config.hidden_size, self.tag_size)

#     def forward(self, batch_data):
#         words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
#         return self.layer_output(words_hidden)

#     def forward_loss(self, batch_data, labelss, ignore_idx=-1):
#         probs = torch.softmax(self(batch_data), dim=-1).clamp(min=1e-9)
#         loss = F.nll_loss(torch.log(probs.transpose(1, 2)), labelss, ignore_index=ignore_idx)
#         # return loss, torch.argmax(probs, dim=-1)
#         return loss

#     def predict(self, batch_data: dict):
#         return torch.argmax(self(batch_data), dim=-1)

#     def load_pretrained(self, pretrained_model_name_or_path):
#         return self.layer_bert.load_pretrained(pretrained_model_name_or_path)

#     def fix_bert(self):
#         return self.set_layer_trainable('layer_bert', False)

#     def get_params_by_part(self):
#         bert_params = list(filter(lambda p: p.requires_grad, self.layer_bert.parameters()))
#         base_params = list(filter(lambda p: id(p) not in list(map(id, bert_params)) and p.requires_grad, self.parameters()))
#         return bert_params, base_params


# class ModelBert_CRF(ModelBert):
#     def __init__(self, config):
#         super(ModelBert_CRF, self).__init__(config)
#         self.layer_crf = CRF(self.tag_size, batch_first=True)

#     def forward_loss(self, batch_data, labelss, ignore_idx=-1):
#         feats = self(batch_data)
#         seq_mask = sequence_mask(batch_data['lengths']).float()
#         log_likelihood = self.layer_crf.forward(feats, labelss, mask=seq_mask.byte(), reduction='mean')
#         loss = -log_likelihood
#         return loss

#     def predict(self, batch_data):
#         feats = self(batch_data)
#         seq_mask = sequence_mask(batch_data['lengths']).float()
#         b_tag_seq = self.layer_crf.decode(feats, mask=seq_mask.byte())
#         return b_tag_seq


class ModelBert_Cofe(ExpModelBase):
    def __init__(self):
        super(ModelBert_Cofe, self).__init__()
        self.tag_size = 7#config['tag_size']
        self.words_dropout_prob = 0.5#config['words_dropout_prob']

        self.layer_bert = WordBert({
                                    "architectures": [
                                        "BertForMaskedLM"
                                    ],
                                    "attention_probs_dropout_prob": 0.1,
                                    "finetuning_task": None,
                                    "hidden_act": "gelu",
                                    "hidden_dropout_prob": 0.1,
                                    "hidden_size": 768,
                                    "initializer_range": 0.02,
                                    "intermediate_size": 3072,
                                    "layer_norm_eps": 1e-12,
                                    "max_position_embeddings": 512,
                                    "model_type": "bert",
                                    "num_attention_heads": 12,
                                    "num_hidden_layers": 12,
                                    "num_labels": 2,
                                    "output_attentions": False,
                                    "output_hidden_states": False,
                                    "pad_token_id": 0,
                                    "pruned_heads": {},
                                    "torchscript": False,
                                    "type_vocab_size": 2,
                                    "vocab_size": 30522
                                    })

        self.words_dropout = nn.Dropout(self.words_dropout_prob)
        self.layer_enh = EnhancedCell({
                                        "tag_size": 7,
                                        "random_seed": 2021,
                                        "true_pred_rand_prob": [0.5, 0.4, 0.1],
                                        "hidden_act": "gelu",
                                        "num_pre_preds": 1,
                                        "num_pre_tokens": 3,
                                        "num_nxt_tokens": 3,
                                        "in_hidden_size": 768,
                                        "hidden_size": 100,
                                        "pred_embedding_dim": 100,
                                        "fc_hy_dropout_prob": 0.5,
                                        "fc_hc_dropout_prob": 0.5,
                                        "fc_hf_dropout_prob": 0.5,
                                        "fc_hl_dropout_prob": 0.5
                                        })

    def forward_loss(self, batch_data, labelss, ignore_idx=-1):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        words_hidden = self.words_dropout(words_hidden)
        loss = self.layer_enh.forward(words_hidden, batch_data['lengths'], labelss, ignore_index=ignore_idx)
        return loss

    def predict(self, batch_data: dict, output_weight=False, output_Z=False):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        outputs = self.layer_enh.predict(words_hidden, batch_data['lengths'], output_weight=output_weight, output_Z=output_Z)
        if isinstance(outputs, tuple):
            return (torch.argmax(outputs[0], dim=-1), ) + outputs[1:]
        else:
            return torch.argmax(outputs, dim=-1)

    def predict_bs(self, batch_data: dict, beam_width=None):
        words_hidden = self.layer_bert(batch_data['tkidss'], batch_data['attention_mask'], batch_data['wdlens'])
        return self.layer_enh.predict_bs(words_hidden, batch_data['lengths'], beam_width)

    def load_pretrained(self, pretrained_model_name_or_path):
        return self.layer_bert.load_pretrained(pretrained_model_name_or_path)

    def fix_bert(self):
        return self.set_layer_trainable('layer_bert', False)

    def get_params_by_part(self):
        bert_params = list(filter(lambda p: p.requires_grad, self.layer_bert.parameters()))
        base_params = list(filter(lambda p: id(p) not in list(map(id, bert_params)) and p.requires_grad, self.parameters()))
        return bert_params, base_params
