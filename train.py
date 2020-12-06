# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018


import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sqlova.utils.utils_wikisql import *
from sqlova.utils.utils import load_jsonl
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=False, action='store_true')   #是否进行训练（无论是否有预训练模型）
    parser.add_argument('--do_infer', default=False, action='store_true')   #是否根据模型进行推测
    parser.add_argument('--infer_loop', default=False, action='store_true') #推断问题个数

    parser.add_argument("--trained", default=False, action='store_true')    #是否有预训练好的模型，没有就自己建一个

    parser.add_argument('--tepoch', default=200, type=int)                  #do_train的迭代次数
    parser.add_argument("--bS", default=32, type=int,                       #batch_size（data转换成torch接受的数据类型时，一次读取多少个数据）
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,      #积累多少次再进行误差反向传播
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',                                      #如果之前bert已经训练了，是否对它进行微调（否则仅仅微调.pt）注意是在写do_train的情况下
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",                                     #词汇表文件，默认"vacab_(us/ul...).txt"
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",                                 #最大总输入序列长度，长了断言报错，短了填充
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",                              #bert的最后层数
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')    #bert的学习率
    parser.add_argument('--seed',                                           #随机初始化值（0-42）
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')  #是否使用bert预训练模型
    parser.add_argument("--bert_type_abb", default='uS', type=str,          #bert模型的默认值（US,UL...）
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, cH, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")     #LSTM的层数
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")              #正则化率
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")            #学习率（LSTM+attention）
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")
                                                                                            #在进行column_attention的时候，隐藏层的个数
    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',                                             #进行一些特殊测试加强性能
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',                                      #？？不知道
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12',
                         'cH': 'chinese_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    # if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
    #     args.do_lower_case = False
    # else:
    #     args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args

# 生成bert模型
def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config.json')      #bert的配置文件
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab.txt')                   #bert的词汇文件
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model.bin')      #bert的预训练模型(不一定有)

    """
    ==BertConfig==该类在bert文件里的modeling里，用bert的配置文件初始化（默认uS）
    <from_json_file>方法用于读取bert配置文件的内容
    """
    bert_config = BertConfig.from_json_file(bert_config_file)

    """
    ==tokenization==bert里的文件
    ==FullTokenizer==类，里面存放词汇信息
    """
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    #毫无作用的输出参数
    bert_config.print_status()

    """
    ==BertModel==该类在bert文件里的modeling里，同样用bert的配置文件初始化，里面有一系列对bert模型的操作（例如添加层，加载参数等...）
    """
    model_bert = BertModel(bert_config)

    if no_pretraining:  #如果不用bert预训练模型，只要它们团队的模型（不需要.bin）
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))     #加载.bin文件，即加载预训练参数
        print("Load pre-trained parameters.")
    model_bert.to(device)

    #      bert模型       词汇       bert配置文件
    return model_bert, tokenizer, bert_config


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    #           0    1      2       3        4     5
    agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']     #agg聚合函数
    #            0    1    2    3
    cond_ops = ['>', '<', '=', '<>']                        # where opt

    print(f"accumulate_gradients = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"bert learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")              #如果之前bert已经训练了，是否对它进行微调（否则仅仅微调.pt）

    # 获取BERT模型（获取bert模型、词汇表、bert配置文件）
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, False,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # bert输出===>seq-to-sql输入


    # 获取Seq-to-SQL模型
    n_cond_ops = len(cond_ops)  #条件长度
    n_agg_ops = len(agg_ops)    #聚合长度
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    '''
    ==Seq2SQL_v1==在sqlova文件夹下，是python源代码Module的子类，里面就是所有层 SC, SA, WN, WC, WO, WV，里面每层都用两层LSTM生成
    参数：输入大小；s2s隐藏层大小(初始化)；LSTM层数；dropout；条件长度；聚合长度
    '''
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    #如果没有预训练就这样返回：只有架子的s2s，有或无架子的bert(取决于写没写no_pretraining)，词汇表，bert配置文件
    #有训练就这样返回：注入参数的s2s，注入参数的bert，词汇表，bert配置文件

    if trained:         #如果模型以及被预训练（trained表示他一定有s2s的pt，no_pretraining表示不需要bert的训练参数，用s2s的就行）
        assert path_model_bert != None
        assert path_model != None

        #用res获取预训练的bert模型参数
        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        #为bert模型注入参数
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        # 用res获取预训练的s2s模型参数
        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')
        #为s2s模型注入参数
        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config


# 获取数据
# 训练数据
# train_data: 训练的所有问题(已分词)，以及对应正确的sql语句     长度：56355
# train_table:训练数据对应的所有表
# dev_data/dev_table: 同train                            长度：8421
# train_loader: 里面存储了训练的所有问题(train_data)，还有batch_size，数据长度等
# dev_loader: 同train
def get_data(path_wikisql, args):
    #注意这里no_hs_tok是说table有没有分词，而不是问题，问题都默认分词了
    #还要注意的是，他还有一个wvi_corenlp字段
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
    model.train()       #将模块设置为训练模式/评估模式。
    model_bert.train()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_wv = 0  # of where-value
    cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0  # of execution acc

    # 初始化数据库查询引擎
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):
        '''t为batch_size个数据'''

        cnt += len(t)   #每个循环中的例子数

        if cnt < st_pos:
            continue
        # 将问题拆分成一个个部分，并且结合问题所对应的表
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu：bs个问题
        # nlu_t：标记化的问题，这里不分词
        # sql_i：SQL查询的规范形式
        # sql_q：完整的SQL查询文本。 不曾用过。
        # sql_t：没软用
        # tb：bs个问题对应的表格（不一定一对一，但是保证bs个问题要找的表在里面）
        # hs_t：标记化的标头。 不曾用过。
        # hds：表头

        '''分别获取bs个问题的sc, sa, wn, wc, wo, wv(多个wn都放在list里)'''
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        '''这个是获取loader里WV的起止(有问题待改进)'''
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        '''函数作用：获取所有从bert模型中输出的参数'''
        # wemb_n: 问题的参数
        # wemb_h: 表字段的参数
        # l_n: 问题的长度
        # l_hpu: 我们不是把问题和表头合在一起了嘛，这就是通过表头的起始，获取每个表头字段的长度
        # l_hs: 表字段总数
        # nlu_tt: 已经分词了的问题
        # t_to_tt_idx: 将已分词的每个字(词)标记它的序号
        # tt_to_t_idx: 同上？

        try:
            #验证/过滤？
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        # 上面已经获取了bert模型的输出，这里将这个输出输入到s2s模型中（并结合问题json的各个字段），获取这个模型得出的bat_sizen内六大关键元素的权重
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi)

        # 生成/计算损失值
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        # 计算梯度
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # 预测得出：最可能的sc/sa/wn/wc/wo/wvi
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        # 根据预测得出的wv起始位置来获取where-value值
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)  #对预测出的wc进行排序
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu) #由预测出的pr_等生成对应的sql语句表示

        # 计算准确率（1：正确 0：错误）
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='train')
        #是否全对/全对的数量（完美的sql，全对：1 不全对：0）
        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # 获得结果，出现小错误频率大王！！
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out


def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx / cnt:.3f}, acc_x = {cnt_x / cnt:.3f}\n',
          f'acc_sc = {cnt_sc / cnt:.3f}, acc_sa = {cnt_sa / cnt:.3f}, acc_wn = {cnt_wn / cnt:.3f}\n',
          f'acc_wc = {cnt_wc / cnt:.3f}, acc_wo = {cnt_wo / cnt:.3f}, acc_wv = {cnt_wv / cnt:.3f}')
    print(f'===============================')


def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []
    #初始化数据库查询引擎
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []    #相比于train，他就多了个收取结果的数组
    for iB, t in enumerate(data_loader):
        '''t为bs个数据的详情'''
        cnt += len(t)   #每个循环中的例子数
        if cnt < st_pos:
            continue
        # 将问题拆分成一个个部分，并且结合问题所对应的表
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        # nlu：bs个问题
        # nlu_t：标记化的问题，这里不分词
        # sql_i：SQL查询的规范形式
        # sql_q：完整的SQL查询文本。 不曾用过。
        # sql_t：没软用
        # tb：bs个问题对应的表格（不一定一对一，但是保证bs个问题要找的表在里面）
        # hs_t：标记化的标头。 不曾用过。
        # hds：表头

        '''分别获取bs个问题的sc, sa, wn, wc, wo, wv(多个wn都放在list里)'''
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        '''这个是获取loader里WV的起止'''
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        '''函数作用：获取所有从bert模型中输出的参数'''
        # wemb_n: 问题的参数
        # wemb_h: 表字段的参数
        # l_n: 问题的长度
        # l_hpu: 我们不是把问题和表头合在一起了嘛，这就是通过表头的起始，获取每个表头字段的长度
        # l_hs: 表字段总数
        # nlu_tt: 已经分词了的问题
        # t_to_tt_idx: 将已分词的每个字(词)标记它的序号
        # tt_to_t_idx: 同上？


        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # 上面已经获取了bert模型的输出，这里将这个输出输入到s2s模型中（并结合问题json的各个字段），获取这个模型得出的bat_sizen内六大关键元素的权重
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # 生成/计算损失值
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # 预测得出：最可能的sc/sa/wn/wc/wo/wvi
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            # 根据预测得出的wv起始位置来获取where-value值
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            # 由预测出的pr_等生成对应的sql语句表示
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None
            loss = torch.tensor([0])

        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sql_i, pr_sql_i,
                                                      mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


def tokenize_corenlp(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1):
        for tok in sentence:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    # for sentence in client.annotate(nlu1).sentence:
    #     for tok in sentence.token:
    #         nlu1_tok.append(tok.originalText)
    '''2020/12/02修改：infer分词函数'''
    # nlu1_tok = list(jieba.cut(nlu1))
    # return nlu1_tok
    ann = client.annotate(nlu1)
    sentence = ann.sentence[0]
    for token in sentence.token:
        nlu1_tok.append(token.word)
    #print(nlu1_tok)
    return nlu1_tok


def infer(nlu1,
          table_name, data_table, path_db, db_name,
          model, model_bert, bert_config, max_seq_length, num_target_layers,
          beam_size=4, show_table=False, show_answer_only=False):
    # I know it is of against the DRY principle but to minimize the risk of introducing bug w, the infer function introuced.
    model.eval()
    model_bert.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    # 问题输入
    nlu = [nlu1]    #问题数组
    '''
    ==tokenize_corenlp_direct_version函数作用：就是英文分词(可能按照stanza规则分?)==
    ==client:stanford的corenlp代理类==
    ==nlu1:刚刚定义的问题列表==
    2020/12/02修改：修改infer中文分词问题
    '''
    nlu_t1 = tokenize_corenlp_direct_version(client, nlu1)
    nlu_t = [nlu_t1]    # 把分词之后的数据也放到数组里

    #tb1 = data_table[0]
    '''
        2020/12/01修改：tb1根据问题来选择表
        循环查找即可
    '''
    for temple_table in data_table:
        if temple_table['name'] == table_name:
            tb1 = temple_table
            break
    hds1 = tb1['header']
    tb = [tb1]
    hds = [hds1]
    hs_t = [[]]
    # 获取bert-output
    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
    # 获取sqlova-output
    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, engine, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size)

    # 切分出where-col/where-op/where-val
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    if len(pr_sql_i) != 1:  # 判断是不是生成了conds
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])     # 根据上面的conds生成sql语句
    pr_sql_q = [pr_sql_q1]      # 将生成的sql语句放到list里，因为infer可能有多句

    '''下面执行SQL语句'''
    try:
        pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
    except:
        pr_ans = ['Answer not found.']
        pr_sql_q = ['Answer not found.']

    if show_answer_only:
        print(f'Q: {nlu[0]}')
        print(f'A: {pr_ans[0]}')
        print(f'SQL: {pr_sql_q}')

    else:
        print(f'START ============================================================= ')
        print(f'{hds}')
        if show_table:
            print(engine.show_table(table_name))
        print(f'nlu: {nlu}')
        print(f'pr_sql_i : {pr_sql_i}')
        print(f'pr_sql_q : {pr_sql_q}')
        print(f'pr_ans: {pr_ans}')
        print(f'---------------------------------------------------------------------')

    return pr_sql_i, pr_ans


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


if __name__ == '__main__':

    ## 1. 超参数
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './data_and_model'  # '/home/wonseok'
    path_wikisql = './data_and_model'  # os.path.join(path_h, 'data', 'wikisql_tok')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. 获取数据

    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )


    ## 4. 没有预训练好的模型就建立模型；有预训练的模型就加载模型（这里是说整个模型，包括bert和接下来的sqlova）
    if not args.trained:
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)
    else:
        # To start from the pre-trained models, un-comment following lines.
        path_model_bert = './data_and_model/model_bert_best.pt'
        path_model = './data_and_model/model_best.pt'
        model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True,
                                                               path_model_bert=path_model_bert, path_model=path_model)



    ## 5. Get optimizers
    if args.do_train:
        #对s2s层的操作集合，对bert的操作集合(如果不fine-tune就不对bert模型进行操作)
        opt, opt_bert = get_opt(model, model_bert, args.fine_tune)

        ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1
        for epoch in range(args.tepoch):
            #train
            acc_train, aux_out_train = train(train_loader,
                                             train_table,
                                             model,
                                             model_bert,
                                             opt,
                                             bert_config,
                                             tokenizer,
                                             args.max_seq_length,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0,
                                             path_db=path_wikisql,
                                             dset_name='train')

            # check DEV
            with torch.no_grad():
                acc_dev, results_dev, cnt_list = test(dev_loader,
                                                      dev_table,
                                                      model,
                                                      model_bert,
                                                      bert_config,
                                                      tokenizer,
                                                      args.max_seq_length,
                                                      args.num_target_layers,
                                                      detail=False,
                                                      path_db=path_wikisql,
                                                      st_pos=0,
                                                      dset_name='dev', EG=args.EG)

            print_result(epoch, acc_train, 'train')
            print_result(epoch, acc_dev, 'dev')

            # save results for the official evaluation
            save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

            # save best model
            # Based on Dev Set logical accuracy lx
            acc_lx_t = acc_dev[-2]
            if acc_lx_t > acc_lx_t_best:
                acc_lx_t_best = acc_lx_t
                epoch_best = epoch
                # save best model
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join('.', 'model_best.pt'))

                state = {'model_bert': model_bert.state_dict()}
                torch.save(state, os.path.join('.', 'model_bert_best.pt'))

            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")

    if args.do_infer:
        # To use recent corenlp: https://github.com/stanfordnlp/python-stanford-corenlp
        # 1. pip install stanford-corenlp
        # 2. download java crsion
        # 3. export CORENLP_HOME=/Users/wonseok/utils/stanford-corenlp-full-2018-10-05

        # from stanza.nlp.corenlp import CoreNLPClient
        # client = CoreNLPClient(server='http://localhost:9000', default_annotators='ssplit,tokenize'.split(','))

        import corenlp

        client = corenlp.CoreNLPClient(annotators='ssplit,tokenize'.split(','))

        '''2020/12/02修改：infer分词函数'''
        # from nltk.tokenize.stanford import CoreNLPTokenizer
        # sttok = CoreNLPTokenizer('http://localhost:9000')   #注意端口号要对应启动stanza的端口号
        # import jieba

        # nlu1 = "长沙2011年平均每天成交量是3.17，那么近一周的成交量是多少"
        # path_db = 'data_and_model'
        # db_name = 'dev'
        # data_table = load_jsonl('./data_and_model/dev.tables.json')
        # table_name = 'Table_69cc8c0c334311e98692542696d6e445'
        nlu1 = "请问股票代码为831155的英文名称是什么？"
        path_db = 'data_and_model'
        db_name = 'sqlova_ch'
        data_table = load_jsonl('data_and_model/com_message/com_message.tables.json')
        table_name = 'Table_com_message'
        n_Q = 100000 if args.infer_loop else 1
        for i in range(n_Q):
            if n_Q > 1:
                nlu1 = input('Type question: ')
            pr_sql_i, pr_ans = infer(
                nlu1,
                table_name, data_table, path_db, db_name,
                model, model_bert, bert_config, max_seq_length=args.max_seq_length,
                num_target_layers=args.num_target_layers,
                beam_size=1, show_table=False, show_answer_only=False
            )
