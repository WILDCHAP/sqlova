## sqlova_ch_base
sqlova中文-基于TableQA天池NL2SQL数据集实现

致谢：  
code-base: [sqlova](https://github.com/naver/sqlova)  
data-base: [TableQA](https://github.com/ZhuiyiTechnology/TableQA)  


基础修改：
* 添加一种bert_type_abb：chinese_L-12_H-768_A-12
* 下载谷歌[chinese-base](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)跑出pytorch-bin
* 修改load_data里，jsonl->json，先不用分词，并且修改数据打开方式，否则文件读取gbk错误（这里表名是name字段不是id字段，但是list名要设成id，为了之后训练）
* 修改model里的get_bert（注意cH没有do_lowercase属性）
* 由于我们没有分词，所以要将train里的question_tok字段换成每个词（get_fields_1里）；sql和query一样；sql_i要注意中文数据里的agg里面还分了一个字段，要么在这改要么在get_g里改；还有这里相比英文少了一个wvi_corenlp(这个是annotation_ws生成的，代表WV的起始和结束位置，我们自己写一个函数找到他，叫ch_corenlp，我们不修改原数据，只在从硬盘读入的时候进行查找（待优化）)（在查找的时候，我们要注意比如10会写成十这样的，还有词会分开来的，非常多情况！！重点必须改进）（2020/11/28改进：直接忽略查不到WV的记录）
* 修改tain里获取每个bs问题的sc，sa等，里面字段cond_conn_op我们先不用，像他英文集合一样先全默认and
* wikimodel中有个是二维，不是三维（pr_sc的问题，因为天池数据集是查找一列或者两列，而sqlova只查找一列所以g_sc在sqlova中是一维list，天池是二维list）这里我们只将要选取的第一列作为结果（待改进）（2020/11/28改进：直接忽略查两列的记录）
* 一些小细节：Table的T大写，col_3，op和agg的含义不同，并且不等于操作要改为<>，数据库里是从1开始的，但是数据集里是从0开始的（将数据集里读取的值全+1）


### 2020/11/28更新：
在对数据进行预处理时，将：select-col数量为2列的、where-value未出现在问题之中的、where之间连接条件是or的，通通舍弃掉（操作位置：utils_wikisql.py-line78）

### 2020/11/29更新：
test函数的更改类似train函数，但是要注意（大概utils_wikisql-line1877），中文agg和sel是[0]和[1]，wikisql是0，1，也就是说要直接从数组中提取（这里还找到了一个agg不是list，为了以防万一还是加一个）


### 2020/11/30成果：
跑出了一个epoch，base-file==11/29版train，继续跑8个epoch


### 2020/12/01更新：
1. 对数据库方法进行修改，之前只修改了execute方法，把剩下的方法也按照它一样修改（dbengine.py-linedbengine）
2. 发现Infer函数问题：(train-line656)源代码直接指定第一张表(tb1 = data_table[0])，并修改（这里其实还存在一个问题：如果找不到这张表怎么办，其实只需要加一个简单的判断，但这里暂时没加）
（下阶段目标：尝试哈工大bert、分词[效果好可以不用分词]、研究在stanza上跑中文模型[infer需要stanza中文]）


### 2020/12/02更新：
1.将昨天改好的dbengine.py和train.py更新到服务器；  
2.跑出哈工大bert的pytorch版并适用于sqlova(转失败了，还是乖乖用官方提供的转好的把，但是好像还是用不了哈工大的，待解决)  
3.利用跑的epoch做infer，这里服务器装的pytorch是1.7.0，改一下windows上的版本  
4.在服务器上安装jdk1.8和stanford包，要在git上clone下来修改client.py，而且要在浏览器中启动服务  

**发现问题**：  
在train.py-line653左右，对问题进行分词出错，调试英文也出现同样的问题，考虑stanford问题，重新换英文启动命令，发现执行正常，进一步证明是中文斯坦福jar包的split出错  
**解决方法**：  
按照一个字一个字分，自己写，注意28和28.0都只分出一个，import nltk没效果，尝试jieba分词，有效果但效果一般，最后用回corenlp  
**修改了**：  
· utils_wikisql.py-line1877和1881，把原来的int换成了int64，同时修改了主文件（WV）和哈工大bert的，没修改服务器的  
· dbengine.py里面所有执行代码前判断Not star with table->Table，for循环里按照','分隔而不是', '，同时修改了主文件（WV）和哈工大bert的，没修改服务器的  
· train.py按照上面的解决方案修改了Infer部分对问题进行分词的部分，同时修改了主文件（WV）和哈工大bert的，没修改服务器的  
· wikisql_models.py-line277，将'agg': 0, 'sel': 1修改成'agg': [0], 'sel': [1]（TableQA格式），同时修改了主文件（WV）和哈工大bert的，没修改服务器的  
· sqlova/utils/utils.py-line75，TableQA之前都是默认一张表占一行，然后一个.table.json文件多张表，由于我们只有一个文件一张表，且不是一行，后面直接把table变成一行==所以这个相当于没改  

最后总体在infer表现还行，对于小数和and条件不敏感  
还创建了con_message（表文件名），sqlova_ch.db（数据库名），注意在创建con_message表的时候，表名为Table_con_message（表名），列名为col_1 -- col_18，类型为text且长度不填  

### 2020/12/03更新：
要解决debengine的sql字段问题  
**主要原因**：sqlite建表语句有引号  
**解决思路**：先把这张表复制到test.db中，再打开，发现建表语句还是有引号，而且records源码并未异常，证明问题主要出在表上；再尝试把test.db表复制过来改结构加字段，还是不行；再尝试sqlite2创建表，还是不行；发现几个表的DDL不同；最后换一个sqlite工具打开了。  

要修改predict.py  
**设想**：在infer版上改，然后直接把predict复制过去  
**修改中**...  
1. predict.py-line122，线程数改为0
2. 初始参数除了他给的以外，建议还加上max_seq_length 280, bS 16
3. 修改了utils_wikisql.py-line1750，将'agg': 0, 'sel': 1修改成'agg': [0], 'sel': [1]（TableQA格式），同时修改了主文件（WV）和哈工大bert的，没修改服务器的
4. predict.py-line103，他原本只生成了sql语句，并没有执行，我这里再把执行结果写进去，虽然sel-col只有一列，但是我还是像agg、sel字段那样，放到了list里
5. 修改了dbengine.py-line117，由于我们要执行sql语句，他原本没有写直接执行的(为了安全考虑还是不再写了)，而是根据conds执行的，我们把原本处理英文的内容注释掉，免得它影响结果，同时修改了主文件（WV）和哈工大bert的，没修改服务器的
6. 修改了predict.py多部分，new了一个compare_sql.py文件，专门用来比较两个sql（不是sql语句）是否相等，然后在predict中调用他（注意比较的时候忽略cond_conn_op）还有特别重要一点是：conds并不是按顺序排的，可能多个where条件顺序是不一样的，判断的时候要按照where-col排序
7. predict.py-line121，增加了一条输出语句，每100个bS也就是1600条输出一次
8. 修改了utils_wikisql.py-line1756/1764，jsonl->json，同时修改了主文件（WV）和哈工大bert的，没修改服务器的
9. sqlova_ch.db和com_message.db是同一个数据库
10. 增加了一个agrs：have_sql，默认为False，如果为False就说明开始不给答案（类似TableQA的test），True的话就会给出准确率，也会给出答案
更新了：
服务器的train.py、wikisql_models.py、utils.py、utils_wikisql.py、dbengine.py
晚上跑表数据，结果惨不忍睹，一行一行查看后，将acc评判标准修改为sql+结果来评判

### 2020/12/04：
将财报表拆分成三个小表，并将格式改为TableQA的格式；观察服务器内存情况（发现内存溢出）
试着拿报表数据做训练或fine-tune
**拆分中**：
1. 放Table_financial_statements表到train.db/dev.db/test.db中，分别250w、150w、100w条		--db
2. 把Table_financial_statements表转成一行放到*.tables.json中					--表
3. 问题（ing...）  
**内存分析**：
在服务器中新建一个复制主文件的，进行内存分析

### 2020/12/05：
写在无sql语句的情况下做infer；将问题代入train/dev/test.json中；查看服务器train函数为什么用了40gb内存；试着用colab跑
1. 完成了未知答案的predict(只修改了predict)；要注意的是，已知答案我们还可以主动排除wv、sc、and/or错误，但这个我们无法排除，所以问问题的时候就不能问这种问题
2. 发现了一个巨大问题：wikisql_models.py-line28初始化max_wn=4，这样同时指定了每张表最多四列，如果由张表只有3列就会报错，除了服务器，其他地方都修改成了3

### 2020/12/06：
将financial_statements的单独问题合并到train/dev中，修改了test的格式(只剩question和tableid)；
测试了financial_statements的单独问题：train/dev准确率很低

### 2020/12/08：
在原6次迭代基础上跑两次迭代，TableQA+fin基础上训练，最终准确率86%，继续跑迭代
修改了predict里每20个bs输出的语句，（iB + 1） * args.bS
用上面训练出来的参数预测单个fin，dev准确率：1613  /  1616
修改了have_sql参数问题，以后不加--have_sql证明为False，加则为True