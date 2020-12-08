#!/usr/bin/env python

# Use existing model to predict sql from tables and questions.
#
# For example, you can get a pretrained model from https://github.com/naver/sqlova/releases:
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_bert_best.pt
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_best.pt
#
# Make sure you also have the following support files (see README for where to get them):
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt
#
# Finally, you need some data - some files called:
#    - <split>.db
#    - <split>.jsonl
#    - <split>.tables.jsonl
#    - <split>_tok.jsonl         # derived using annotate_ws.py
# You can play with the existing train/dev/test splits, or make your own with
# the add_csv.py and add_question.py utilities.
#
# Once you have all that, you are ready to predict, using:
#   python predict.py \
#     --bert_type_abb uL \       # need to match the architecture of the model you are using
#     --model_file <path to models>/model_best.pt            \
#     --bert_model_file <path to models>/model_bert_best.pt  \
#     --bert_path <path to bert_config/vocab>  \
#     --result_path <where to place results>                 \
#     --data_path <path to db/jsonl/tables.jsonl>            \
#     --split <split>
#
# Results will be in a file called results_<split>.jsonl in the result_path.

import argparse, os
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils import load_jsonl
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models, tokenize_corenlp_direct_version
from compare_sql import compare, compare_by_result
# This is a stripped down version of the test() method in train.py - identical, except:
#   - does not attempt to measure accuracy and indeed does not expect the data to be labelled.
#   - saves plain text sql queries.
#

def predict(data_loader, data_table, model, model_bert, bert_config, tokenizer,
            max_seq_length,
            num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
            path_db=None, dset_name='test'):
    '''
    2020/12/03修改：
    right_ans: 正确个数
    '''
    right_ans = 0

    model.eval()
    model_bert.eval()
    # 数据库名字也是取决于split
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):
        '''
        每次处理bS个数据
        nlu:问题      nlu_t:分词了的问题
        sql_i/q:conds  
        tb:bS张表
        hds:bS张表的字段
        '''
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        '''进一步由conds细分出6大部分(注意这里都是已经给出的正确答案，并没有做出预测操作)'''
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        '''获取where-value起止点'''
        g_wvi_corenlp = get_g_wvi_corenlp(t)
        '''获取bert输出'''
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        '''真正predict在以下'''
        if not EG:
            # 获取sqlova得出的6大重要部分参数
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)
            # 将权重参数变成值
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            # 根据值得出where-value(这里是分词版，一个一个切分)
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # 最后在将值整合成conds(pre版本)
            '''2020/12/03修改：将agg和sel变为列表形式（generate_sql_i函数内修改）'''
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
            # Following variables are just for consistency with no-EG case.
            pr_wvi = None # not used
            pr_wv_str=None
            pr_wv_str_wp=None
        '''2020/12/03修改：对比bS个sql，得出准确率'''
        #right_ans += compare(sql_i, pr_sql_i)

        # 得出sql查询语句（其实最后查询还是按照sql查）
        pr_sql_q = generate_sql_q(pr_sql_i, tb)
        # 每次遍历一条（当前从bS个数据中一条条抽取）conds和sql语句，放到结果中
        for b, (pr_sql_i1, pr_sql_q1) in enumerate(zip(pr_sql_i, pr_sql_q)):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results1["sql"] = pr_sql_q1
            '''2020/12/03修改：连接数据库进行查询操作'''
            '''2020/12/03晚上修改：再查出真实值，对比得出准确率'''
            try:

                pr_ans, _ = engine.execute_return_query(tb[b]['id'], pr_sc[b], pr_sa[b], pr_sql_i[b]['conds'])
                ans, _ = engine.execute_return_query(tb[b]['id'], g_sc[b], g_sa[b], sql_i[b]['conds'])
            except:
                pr_ans = ['Answer not found.']
            results1["result"] = pr_ans
            results.append(results1)

            '''得出准确率(这里是先看agg, conds，再比较sel和ans)'''
            if compare(sql_i[b], pr_sql_i[b], ans, pr_ans) == True:
                right_ans += 1

        if iB % 20 == 0:
            print("now position:", iB * args.bS)
            if iB != 0:
                # print acc
                print("=========================================================")
                print("acc：", right_ans, " / ", (iB+1) * args.bS, " ==> ", right_ans / ((iB+1) * args.bS))
                print("=========================================================")
    return results, right_ans

'''predict_nosql参数说明'''
# nlu_list：问题list
# table_name_list：问题对应表名list
# data_table：table数据list
# path_db：数据库文件所在文件夹
# db_name：数据库名字
def predict_nosql(nlu_list,
                  table_name_list, data_table, path_db, db_name,
                  model, model_bert, bert_config, max_seq_length, num_target_layers,
                  beam_size=4, show_table=False, show_answer_only=False):
    # 模型、数据库准备
    model.eval()
    model_bert.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    results = []

    # 一条一条查(也可以修改接口换成dev_loader的bS处理，但是会影响很多文件)
    for b, (nlu, table_name) in enumerate(zip(nlu_list, table_name_list)):
        nlu_t1 = tokenize_corenlp_direct_version(client, nlu)   # 利用stanford中文分词
        nlu_t = [nlu_t1]  # 把分词之后的数据也放到数组里
        # 循环找出该问题要找的那张表
        for temple_table in data_table:
            if temple_table['name'] == table_name:
                tb1 = temple_table
                break
        hds1 = tb1['header']    # 获取表头
        tb = [tb1]
        hds = [hds1]
        '''开始预测'''
        # 获取bert-output
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            # 获取sqlova-output
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, [nlu],
                                                                                            beam_size=beam_size)
        except:
            # 出现错误改wikisql_models.py-line28初始化where-num数量
            print("error table header len:", len(hds1), "table name:", table_name)
            continue

        # 切分出where-col/where-op/where-val
        pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
        pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])  # 根据上面的conds生成sql语句
        pr_sql_q = [pr_sql_q1]  # 将生成的sql语句放到list里

        '''下面执行SQL语句'''
        try:
            pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
        except:
            pr_ans = ['Answer not found.']
            pr_sql_q = ['Answer not found.']

        '''写结果'''
        results1 = {}
        results1["query"] = pr_sql_i[0]
        results1["table_id"] = tb[0]["id"]
        results1["nlu"] = nlu_list[b]
        results1["sql"] = pr_sql_q1
        results1["result"] = pr_ans

        results.append(results1)

        '''每1000条打印一次'''
        if b % 1000 == 0:
            print("now position:", b)

    return results

## Set up hyper parameters and paths
parser = argparse.ArgumentParser()
# sqlova预训练模型文件data_and_model/model_best.pt
parser.add_argument("--model_file", required=True, help='model file to use (e.g. model_best.pt)')
# bert预训练模型文件data_and_model/model_bert_best.pt
parser.add_argument("--bert_model_file", required=True, help='bert model file to use (e.g. model_bert_best.pt)')
# bert的json文件和.db数据库文件所在文件夹
parser.add_argument("--bert_path", required=True, help='path to bert files (bert_config*.json etc)')
parser.add_argument("--data_path", required=True, help='path to *.jsonl and *.db files')
# 问题json文件和对应.db数据库文件的前缀
parser.add_argument("--split", required=True, help='prefix of jsonl and db files (e.g. dev)')
# 结果文件夹
parser.add_argument("--result_path", required=True, help='directory in which to place results')
# 是否给出sql
parser.add_argument("--have_sql", action='store_true', help='hava sql?')
# 除此之外，还要加上max_seq_length 280, bS 16

args = construct_hyper_param(parser)

BERT_PT_PATH = args.bert_path                   # bert文件路径
path_save_for_evaluation = args.result_path     # 结果存放位置

# Load pre-trained models
path_model_bert = args.bert_model_file          # model_bert_best.pt
path_model = args.model_file                    # model_best..pt
args.no_pretraining = True  # counterintuitive, but avoids loading unused models
# 获取：sqlova模型、bert模型等，和train.py一样
model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

# 加载数据--有答案标注的情况下
if args.have_sql == True:
    # 获取数据
    dev_data, dev_table = load_wikisql_data(args.data_path, mode=args.split, toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)

    dev_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=dev_data,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    # Run prediction
    with torch.no_grad():
        '''2020/12/03修改：返回值由一个变两个（再返回一个right_ans），并且在下面根据right_ans得出总体准确率'''
        results, right_ans = predict(dev_loader,
                          dev_table,
                          model,
                          model_bert,
                          bert_config,
                          tokenizer,
                          args.max_seq_length,
                          args.num_target_layers,
                          detail=False,
                          path_db=args.data_path,
                          st_pos=0,
                          dset_name=args.split, EG=args.EG)

        # print acc
        print("=========================================================")
        print("final acc：", right_ans, " / ", len(dev_data), " ==> ", right_ans / len(dev_data))
        print("=========================================================")

# 加载数据--无答案标注的情况下
else:
    nlu_list = []           # 问题list
    table_name_list = []    # 每个问题对应的表名list

    '''从<split>.json中将问题和表名提取出来'''
    path_question = os.path.join(args.data_path, args.split + '.json')  # 问题文件路径名
    question_list = load_jsonl(path_question)       # 利用utils里load_jsonl方法，的将问题文件加载出来先
    for question in question_list:
        nlu_list.append(question['question'])
        table_name_list.append('Table_' + question['table_id'])     # 注意table_name要加‘Table_'
    assert len(nlu_list) == len(table_name_list)  # 安全起见判断一下两个list长度是否一致

    # '''其他参数设定'''
    path_table = os.path.join(args.data_path, args.split + '.tables.json')
    data_table = load_jsonl(path_table)     # 表信息
    path_db = args.data_path                # 数据库所在文件夹
    db_name = args.split                    # 数据库名字

    # 导入stanford中文分词
    import corenlp
    client = corenlp.CoreNLPClient(annotators='ssplit,tokenize'.split(','))

    '''调用predict_nosql'''
    results = predict_nosql(nlu_list,
                            table_name_list, data_table, path_db, db_name,
                            model, model_bert, bert_config, max_seq_length=args.max_seq_length,
                            num_target_layers=args.num_target_layers,
                            beam_size=1, show_table=False, show_answer_only=False)

    '''由于没有标注，所以只输出成果就行'''
    print("ans in dir: " + path_save_for_evaluation)

# Save results
save_for_evaluation(path_save_for_evaluation, results, args.split)
