import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


# class BigModel(nn.Module):
#     def __init__(self, main_model):
#         super(BigModel, self).__init__()
#         self.main_model = main_model
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, tok, att, cud=True):
#         typ = torch.zeros(tok.shape).long()
#         if cud:
#             typ = typ.cuda()
#         pooled_output = self.main_model(tok, token_type_ids=typ, attention_mask=att)['pooler_output']
#         logits = self.dropout(pooled_output)
#         return logits


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            self.main_model = BertModel.from_pretrained('bert_pretrained/')
        else:
            config = BertConfig(vocab_size=31090, )
            self.main_model = BertModel(config)

        self.dropout = nn.Dropout(0.1)
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        typ = torch.zeros(input_ids.shape).long().to(device)
        output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
        logits = self.dropout(output)
        return logits


if __name__ == '__main__':
    model = TextEncoder()