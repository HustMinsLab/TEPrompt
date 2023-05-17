# -*- coding: utf-8 -*-
# """
import os

from torch.optim import lr_scheduler

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# """

import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import f1_score
from torch.autograd import Variable
from transformers import RobertaTokenizer, AdamW

from load_data import load_data
from model import RoBERTa_MLM
from parameter import parse_args
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test

torch.cuda.empty_cache()
args = parse_args()  # load parameters


# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

# load RoBERTa model
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
new_tokens = ['<cls_arg1>', '<cls_arg2>']
tokenizer.add_tokens(new_tokens)
args.vocab_size = len(tokenizer)

# load data tsv file
train_data, dev_data, test_data = load_data()

# get arg_1 arg_2 label from data
train_arg_1, train_arg_2, train_label, train_conn_label, train_conn, train_examples = prepro_data_train(train_data)
dev_arg_1, dev_arg_2, dev_label, dev_conn, dev_examples = prepro_data_dev(dev_data)
test_arg_1, test_arg_2, test_label, test_conn, test_examples = prepro_data_test(test_data)

# replace the words which can not be convert into one token with virtual words
def change_examples(tokenizer, label_example):
    to_add = {}
    convert_dict = {}
    convert_dict_reverse = {}

    new_label_example = []
    s = 0
    for i in range(len(label_example)):
        connective = ' ' + label_example[i]
        if connective in convert_dict_reverse:
            continue
        # test weather the connective can be convert into one token
        temp = tokenizer(connective)['input_ids'][1: -1]
        if len(temp) > 1:
            to_add['[A-' + str(s) + ']'] = temp
            convert_dict['[A-' + str(s) + ']'] = connective
            convert_dict_reverse[connective] = '[A-' + str(s) + ']'
            s += 1
    tokenizer.add_tokens(list(to_add.keys()))
    args.vocab_size = len(tokenizer)

    result = {}
    result_reverse = {}
    tokens_ids = []
    for l in label_example:
        l = ' ' + l
        if l in convert_dict_reverse:
            l = convert_dict_reverse[l]
        if l not in result:
            t = tokenizer(l)['input_ids'][1]
            result_reverse[t] = len(result)
            result[l] = t
            tokens_ids.append(t)

    for o in label_example:
        o = ' ' + o
        temp = [0] * len(result)
        if o in convert_dict_reverse:
            temp[result_reverse[result[convert_dict_reverse[o]]]] = 1
        else:
            temp[result_reverse[result[o]]] = 1
        new_label_example.append(temp)
    return to_add, new_label_example, tokens_ids


to_add, train_label_examples_index, Token_id_3 = change_examples(tokenizer, train_examples)

label_conn = torch.LongTensor(train_conn_label)
label_tr = torch.LongTensor(train_label)
label_de = torch.LongTensor(dev_label)
label_te = torch.LongTensor(test_label)
label_example = torch.LongTensor(train_label_examples_index)
print('Data loaded')

Comp = ['similarly', 'but', 'however', 'although']
Cont = ['for', 'if', 'because', 'so']
Expa = ['instead', 'by', 'thereby', 'specifically', 'and']
Temp = ['simultaneously', 'previously', 'then']

label_2 = ['comparison', 'contingency', 'expansion', 'temporal']

len_comp = len(Comp)
len_cont = len(Cont)
len_expa = len(Expa)
len_temp = len(Temp)

# corresponding ids of the above connectives when tokenizing
Token_id_1 = [11401, 53, 959, 1712,
              13, 114, 142, 98,
              1386, 30, 12679, 4010, 8,
              11586, 1433, 172]

Token_id_2 = [24584, 2919, 41853, 6676]


# to limit the length of argument_1 and argument_2
def arg_1_prepro(arg_1, arg_2):
    arg_1_new = []
    arg_1_len = []
    arg_2_new = []
    for each_string in arg_1:
        encode_dict = tokenizer.encode_plus(
            each_string,
            add_special_tokens=False,
            padding='max_length',
            max_length=args.arg1_len,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt')
        decode_input = tokenizer.decode(encode_dict['input_ids'][0]).replace('<pad>', '')
        arg_1_new.append(decode_input)
        arg_1_len.append(len(tokenizer.encode_plus(decode_input, add_special_tokens=False,
                                                   max_length=args.arg1_len, truncation=True, pad_to_max_length=False,
                                                   return_attention_mask=False, return_tensors='pt')['input_ids'][0]))
    for o in range(len(arg_2)):
        max_l_2 = args.len_arg - arg_1_len[o] - 33
        encode_dict_2 = tokenizer.encode_plus(
            arg_2[o],
            add_special_tokens=False,
            padding='max_length',
            max_length=max_l_2,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=False,
            return_tensors='pt')
        decode_input_2 = tokenizer.decode(encode_dict_2['input_ids'][0]).replace('<pad>', '')
        arg_2_new.append(decode_input_2)
    return arg_1_new, arg_2_new


train_arg_1, train_arg_2 = arg_1_prepro(train_arg_1, train_arg_2)
dev_arg_1, dev_arg_2 = arg_1_prepro(dev_arg_1, dev_arg_2)
test_arg_1, test_arg_2 = arg_1_prepro(test_arg_1, test_arg_2)

# cls_indices: the location of the last two [CLS]
def get_batch(text_data1, text_data2, indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # the place of '<mask>' in 'input_ids'
    cls_indices = []

    for idx in indices:
        encode_dict = tokenizer.encode_plus(
            # Prompt
            '<cls_arg1> ' + text_data1[idx] + ' </s> ' + ' <mask> ' + '<cls_arg2> ' + text_data2[idx]
            + ' </s> <s> the sense between <cls_arg1> and <cls_arg2> is <mask> </s> <s> the connective word is '
            + '<mask> ',
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        try:
            mask_indices.append(
                [temp[1] for temp in np.argwhere(np.array(encode_dict['input_ids']) == 50264)])  # id of <mask> is 50264
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))

        try:
            cls_indices.append(
                [temp[1] for temp in np.argwhere(np.array(encode_dict['input_ids']) == 0)][1:])  # id of <s> is 0
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))
    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)

    return batch_ids, batch_mask, mask_indices, cls_indices


# ---------- network ----------
net = RoBERTa_MLM(args).cuda()

# update the representation of the virtual words
net.handler(to_add, tokenizer)
# AdamW
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.milestones1, args.milestones2], gamma=args.warm_ratio)
criterion = nn.CrossEntropyLoss().cuda()
# creat file to save model and result
file_out = open('./' + args.file_out + time.strftime('%Y-%m-%d %H-%M-%S'.format(time.localtime())) + '.txt', "w")

print('epoch_num:', args.num_epoch)
print('epoch_num:', args.num_epoch, file=file_out)
print('wd:', args.wd)
print('wd:', args.wd, file=file_out)
print('initial_lr:', args.lr)
print('initial_lr:', args.lr, file=file_out)
print('batch_size:', args.batch_size)
print('batch_size:', args.batch_size, file=file_out)
print('Loss2:', args.Loss2)
print('Loss2:', args.Loss2, file=file_out)
print('Loss3:', args.Loss3)
print('Loss3:', args.Loss3, file=file_out)
print('warm_ratio:', args.warm_ratio)
print('warm_ratio:', args.warm_ratio, file=file_out)


dev_f1_best = 0
test_acc_best = 0
test_f1_best = 0

last_dev_result = 0
dev_result_down_times = 0
##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch + 1)
    print('Epoch: ', epoch + 1, file=file_out)
    all_indices = torch.randperm(args.train_size).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    start = time.time()

    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], file=file_out)

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                         desc='Train {}'.format(epoch))
    for i, batch_indices in enumerate(all_indices, 1):

        progress.update(1)
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices, token_cls_indices = get_batch(train_arg_1, train_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_tr[batch_indices]).cuda()
        y_conn = label_conn[batch_indices].cuda()
        y_conn_2 = label_tr[batch_indices].cuda()
        y_conn_3 = label_example[batch_indices].cuda()

        # fed data into network
        out_sense, out_ans_1, out_ans_2, out_ans_3 = net(batch_arg, mask_arg, token_mask_indices, token_cls_indices,
                                                         Token_id_1, Token_id_2, Token_id_3, len_comp, len_cont,
                                                         len_expa, len_temp)

        _, pred_ans_1 = torch.max(out_ans_1, dim=1)
        _, truth_ans_1 = torch.max(y_conn, dim=1)
        _, truth_ans_2 = torch.max(y_conn_2, dim=1)
        _, truth_ans_3 = torch.max(y_conn_3, dim=1)

        #
        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)

        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

        # loss
        L1 = criterion(out_ans_1, truth_ans_1)
        L2 = criterion(out_ans_2, truth_ans_2)
        L3 = criterion(out_ans_3, truth_ans_3)
        loss = L1 + args.Loss2 * L2 + args.Loss3 * L3
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        loss_epoch += loss.item()
        if i % (3000 // args.batch_size) == 0:
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                             average='macro')), file=file_out)
            print('loss={:.4f}, acc={:.4f}, F1_score={:.4f}'.format(loss_epoch / (3000 // args.batch_size), acc / 3000,
                                                                    f1_score(f1_truth.cpu(), f1_pred.cpu(),
                                                                             average='macro')))
            loss_epoch = 0.0
            acc = 0.0
            f1_pred = torch.IntTensor([]).cuda()
            f1_truth = torch.IntTensor([]).cuda()
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))
    progress.close()
    scheduler.step()
    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.randperm(args.dev_size).split(args.batch_size)
    loss_epoch = []
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()

    net.eval()
    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                         desc='Dev {}'.format(epoch))
    for batch_indices in all_indices:
        progress.update(1)
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices, token_cls_indices = get_batch(dev_arg_1, dev_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_de[batch_indices]).cuda()

        # fed data into network
        out_sense, out_ans_1, _, _, = net(batch_arg, mask_arg, token_mask_indices, token_cls_indices, Token_id_1,
                                         Token_id_2, Token_id_3, len_comp, len_cont, len_expa, len_temp)
        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

    # report
    dev_f1 = f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc / args.dev_size, dev_f1), file=file_out)
    print('Dev Acc={:.4f}, Dev F1_score={:.4f}'.format(acc / args.dev_size, dev_f1))
    progress.close()

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.randperm(args.test_size).split(args.batch_size)
    loss_epoch = []
    acc = 0.0
    f1_pred = torch.IntTensor([]).cuda()
    f1_truth = torch.IntTensor([]).cuda()
    net.eval()
    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                         desc='Test {}'.format(epoch))

    # Just for Multi-Prompt case
    '''
    test_pred = torch.zeros(1474, 4)
    test_truth = torch.zeros(1474, 4)
    '''

    for batch_indices in all_indices:
        progress.update(1)
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices, token_cls_indices = get_batch(test_arg_1, test_arg_2, batch_indices)

        batch_arg = batch_arg.cuda()

        mask_arg = mask_arg.cuda()

        y = Variable(label_te[batch_indices]).cuda()

        # fed data into network
        out_sense, out_ans_1, _, _, = net(batch_arg, mask_arg, token_mask_indices, token_cls_indices, Token_id_1,
                                         Token_id_2, Token_id_3, len_comp, len_cont, len_expa, len_temp)

        # Just for Multi-Prompt case
        # choose the appropriate Prompt(1 2 3) and other parameters
        # save outputs of the model
        # all test_truth are the same
        '''
        # -------------------------
        for i in range(len(batch_indices)):
            test_pred[batch_indices[i]] = out_sense[i]
            test_truth[batch_indices[i]] = y[i]
        if epoch == 5:                        # epoch(variable) + 1 = epoch(real)
            torch.save(test_pred, './RoBERTa_prompt1.pth')
            # torch.save(test_truth, './test_truth.pth')   # once is ok
        # ------------------------
        '''

        _, pred = torch.max(out_sense, dim=1)
        _, truth = torch.max(y, dim=1)
        num_correct = (pred == truth).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, pred), 0)
        f1_truth = torch.cat((f1_truth, truth), 0)

    # report
    test_f1 = f1_score(f1_truth.cpu(), f1_pred.cpu(), average='macro')
    test_acc = acc / args.test_size
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(test_acc, test_f1),
          file=file_out)
    print('Test Acc={:.4f}, Test F1_score={:.4f}'.format(test_acc, test_f1))
    progress.close()

    # Current best results
    if dev_f1 > dev_f1_best:
        dev_f1_best = dev_f1
        test_f1_best = test_f1
        test_acc_best = test_acc

    # Early stop
    if dev_f1 < last_dev_result:
        dev_result_down_times += 1
        if dev_result_down_times >= args.down_times:
            print("Maybe over fitting, early stop now!")
            break
    else:
        dev_result_down_times = 0

print("Dev best f1 score:", dev_f1_best)
print("Test best acc score:", test_acc_best)
print("Test best f1 score:", test_f1_best)
print("Dev best f1 score:", dev_f1_best, file=file_out)
print("Test best acc score:", test_acc_best, file=file_out)
print("Test best f1 score:", test_f1_best, file=file_out)

file_out.close()
