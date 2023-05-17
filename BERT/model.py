# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertForMaskedLM


class BERT_MLM(nn.Module):
    def __init__(self, args):
        super(BERT_MLM, self).__init__()

        self.BERT_MLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.BERT_MLM.resize_token_embeddings(args.vocab_size)
        for param in self.BERT_MLM.parameters():
            param.requires_grad = True
        self.hidden_size = 768

        self.vocab_size = args.vocab_size
        self.num_class = args.num_class

        # gate1
        self.W1_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W1_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # gate2
        self.W2_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, arg, mask_arg, token_mask_indices, token_cls_indices, Token_id_1, Token_id_2, Token_id_3,  len_comp, len_cont,
                len_expa, len_temp):
        out_arg = self.BERT_MLM.bert(arg, attention_mask=mask_arg, output_hidden_states=True)[0].cuda()

        out_hidden_mask = torch.zeros((len(arg), 3, self.hidden_size)).cuda()
        out_hidden_cls_1 = torch.zeros((len(arg), 1, self.hidden_size)).cuda()
        out_hidden_cls_2 = torch.zeros((len(arg), 1, self.hidden_size)).cuda()

        for i in range(len(arg)):
            out_hidden_mask[i] = out_arg[i][token_mask_indices[i]]
            out_hidden_cls_1[i] = out_arg[i][token_cls_indices[i][0]]
            out_hidden_cls_2[i] = out_arg[i][token_cls_indices[i][1]]

        # fuse the feature
        gate_1 = torch.sigmoid(self.W1_1(out_hidden_cls_1) + self.W1_2(out_hidden_cls_2)).cuda()
        out_gate_1 = (torch.mul(gate_1, out_hidden_cls_1) + torch.mul(1.0 - gate_1, out_hidden_cls_2)).cuda()

        temp = out_hidden_mask[:, 0, :].clone().detach().unsqueeze(1)
        gate_2 = torch.sigmoid(self.W2_1(out_gate_1) + self.W2_2(temp)).cuda()
        out_hidden_mask[:, 0, :] = (torch.mul(gate_2, out_gate_1) + torch.mul(1.0 - gate_2, temp)).squeeze(1).cuda()
        out_vocab = self.BERT_MLM.cls(out_hidden_mask)

        out_vocab_1 = out_vocab[:, 0, :]
        out_vocab_2 = out_vocab[:, 1, :]
        out_vocab_3 = out_vocab[:, 2, :]

        out_ans_1 = out_vocab_1[:, Token_id_1]
        out_ans_2 = out_vocab_2[:, Token_id_2]
        out_ans_3 = out_vocab_3[:, Token_id_3]
        pred_word = torch.argmax(out_ans_1, dim=1).tolist()

        pred = torch.IntTensor(len(arg), self.num_class).cuda()
        for tid, idx in enumerate(pred_word, 0):
            if idx <= (len_comp - 1):
                pred[tid] = torch.IntTensor([1, 0, 0, 0])
            elif (len_comp - 1) < idx <= (len_comp + len_cont - 1):
                pred[tid] = torch.IntTensor([0, 1, 0, 0])
            elif (len_comp + len_cont - 1) < idx <= (len_comp + len_cont + len_expa - 1):
                pred[tid] = torch.IntTensor([0, 0, 1, 0])
            elif (len_comp + len_cont + len_expa - 1) < idx <= (len_comp + len_cont + len_expa + len_temp - 1):
                pred[tid] = torch.IntTensor([0, 0, 0, 1])

        return pred, out_ans_1, out_ans_2, out_ans_3

    # update the representation of some words
    def handler(self, to_add, tokenizer):
        w = self.BERT_MLM.bert.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).cuda()
                for j in l:
                    temp += w[j]
                temp /= len(l)

                w[tokenizer.convert_tokens_to_ids(i)] = temp

