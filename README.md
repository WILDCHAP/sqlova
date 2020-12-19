# sqlova_ch_base
sqlova中文-基于TableQA天池NL2SQL数据集实现

------------
**致谢：**  

code-base: [sqlova](https://github.com/naver/sqlova)  
data-base: [TableQA](https://github.com/ZhuiyiTechnology/TableQA)  

------------

## ****基础修改：****
* 添加一种bert_type_abb：chinese_L-12_H-768_A-12
* 下载谷歌[chinese-base](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)跑出pytorch-bin
* 修改load_data里，jsonl->json，先不用分词，并且修改数据打开方式，否则文件读取gbk错误（这里表名是name字段不是id字段，但是list名要设成id，为了之后训练）
* 修改model里的get_bert（注意cH没有do_lowercase属性）
* 由于我们没有分词，所以要将train里的question_tok字段换成每个词（get_fields_1里）；sql和query一样；sql_i要注意中文数据里的agg里面还分了一个字段，要么在这改要么在get_g里改；还有这里相比英文少了一个wvi_corenlp(这个是annotation_ws生成的，代表WV的起始和结束位置，我们自己写一个函数找到他，叫ch_corenlp，我们不修改原数据，只在从硬盘读入的时候进行查找（待优化）)（在查找的时候，我们要注意比如10会写成十这样的，还有词会分开来的，非常多情况！！重点必须改进）
*（2020/11/28改进：直接忽略查不到WV的记录）*
* 修改tain里获取每个bs问题的sc，sa等，里面字段cond_conn_op我们先不用，像他英文集合一样先全默认and
*（2020/12/15改进：可以使用or）*
* wikimodel中有个是二维，不是三维（pr_sc的问题，因为天池数据集是查找一列或者两列，而sqlova只查找一列所以g_sc在sqlova中是一维list，天池是二维list）这里我们只将要选取的第一列作为结果（待改进）
*（2020/11/28改进：直接忽略查两列的记录）
（2020/12/18改进：都作为list读入，且可以处理多个sel,agg）*
* 一些小细节：Table的T大写，col_3，op和agg的含义不同，并且不等于操作要改为<>，数据库里是从1开始的，但是数据集里是从0开始的（将数据集里读取的值全+1）

------------
## Requirements
- `python3.6.9` 
- `PyTorch 1.2.0` 
- `CUDA 10.0.130`
- `matplotlib 2.2.9`
- `records 0.5.3`
- `matplotlib 2.2.9`
- `babel 2.8.0`
- `defusedxml 0.6.0`
- `tqdm 4.51.0`

------------
## Params
- 训练：`python3 train.py --do_train --tepoch 200 --bS 16 --seed 1 --bert_type_abb cH --lr 0.001 --lr_bert 0.00001 --max_seq_length 280`
- fine-tune：`python3 train.py --do_train --tepoch 30 --bS 16 --seed 1 --bert_type_abb cH --lr 0.001 --lr_bert 0.00001 --max_seq_length 280 --trained --fine_tune`
- infer：`python3 train.py --do_infer --seed 1 --bS 16 --accumulate_gradients 2 --bert_type_abb cH --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_length 280 --trained --no_pretraining`
- predict：
    - `python3 train.py --model_file data_and_model/model_best.pt --bert_model_file data_and_model/model_bert_best.pt --bert_path data_and_model --data_path data_and_model --split test --result_path result --max_seq_length 280 --bS 16 --have_sql`
	- `python3 train.py --model_file data_and_model/model_best.pt --bert_model_file data_and_model/model_bert_best.pt --bert_path data_and_model --data_path data_and_model --split test --result_path result --max_seq_length 280 --bS 16`
	- split表示的是你要预测的.db/.json/.tables.json文件的文件名
	- have_sql属性表示你是有标记(得出准确率和结果)还是无标记(仅得出结果)
------------
## 更新日志：
### **\*\*\*\*\*\*\*\*\*\* 2020/11/28 \*\*\*\*\*\*\*\*\*\***

在对数据进行预处理时，将：select-col数量为2列的、where-value未出现在问题之中的、where之间连接条件是or的，通通舍弃掉*（操作位置：utils_wikisql.py-line78）*

### **\*\*\*\*\*\*\*\*\*\* 2020/11/29 \*\*\*\*\*\*\*\*\*\***
test函数的更改类似train函数，但是要注意（大概utils_wikisql-line1877），中文agg和sel是[0]和[1]，wikisql是0，1，也就是说要直接从数组中提取（这里还找到了一个agg不是list，为了以防万一还是加一个）

### **\*\*\*\*\*\*\*\*\*\* 2020/11/30 \*\*\*\*\*\*\*\*\*\***
跑出了一个epoch，base-file==11/29版train，继续跑8个epoch

### **\*\*\*\*\*\*\*\*\*\* 2020/12/01 \*\*\*\*\*\*\*\*\*\***
1. 对数据库方法进行修改，之前只修改了execute方法，把剩下的方法也按照它一样修改（dbengine.py-linedbengine）
2. 发现Infer函数问题：(train-line656)源代码直接指定第一张表(tb1 = data_table[0])，并修改（这里其实还存在一个问题：如果找不到这张表怎么办，其实只需要加一个简单的判断，但这里暂时没加）

### **\*\*\*\*\*\*\*\*\*\* 2020/12/02 \*\*\*\*\*\*\*\*\*\***
1.将昨天改好的dbengine.py和train.py更新到服务器；  
2.跑出哈工大bert的pytorch版并适用于sqlova(转失败了，还是乖乖用官方提供的转好的把，但是好像还是用不了哈工大的，待解决)  
3.利用跑的epoch做infer，这里服务器装的pytorch是1.7.0，改一下windows上的版本  
4.在服务器上安装jdk1.8和stanford包，要在git上clone下来修改client.py，而且要在浏览器中启动服务  
> **发现问题**：   
> 在train.py-line653左右，对问题进行分词出错，调试英文也出现同样的问题，考虑stanford问题，重新换英文启动命令，发现执行正常，进一步证明是中文斯坦福jar包的split出错
> 
> **解决方法**：   
> 按照一个字一个字分，自己写，注意28和28.0都只分出一个，import nltk没效果，尝试jieba分词，有效果但效果一般，最后用回corenlp

**修改了**：  
 - utils_wikisql.py-line1877和1881，把原来的int换成了int64
 - dbengine.py里面所有执行代码前判断Not star with table->Table，for循环里按照','分隔而不是', '
 - train.py按照上面的解决方案修改了Infer部分对问题进行分词的部分
 - wikisql_models.py-line277，将'agg': 0, 'sel': 1修改成'agg': [0], 'sel':[1]（TableQA格式）
 - sqlova/utils/utils.py-line75，TableQA之前都是默认一张表占一行，然后一个.table.json文件多张表，由于我们只有一个文件一张表，且不是一行，后面直接把table变成一行==所以这个相当于没改
 - 最后总体在infer表现还行，对于小数和and条件不敏感
 - 创建了con_message（表文件名），sqlova_ch.db（数据库名），注意在创建con_message表的时候，表名为Table_con_message（表名），列名为col_1--col_18，类型为text且长度不填

### **\*\*\*\*\*\*\*\*\*\* 2020/12/03 \*\*\*\*\*\*\*\*\*\***
 - 要解决debengine的sql字段问题
**主要原因**：sqlite建表语句有引号  
**解决思路**：先把这张表复制到test.db中，再打开，发现建表语句还是有引号，而且records源码并未异常，证明问题主要出在表上；再尝试把test.db表复制过来改结构加字段，还是不行；再尝试sqlite2创建表，还是不行；发现几个表的DDL不同；最后换一个sqlite工具打开了。  
 - 要修改predict.py
**设想**：在infer版上改，然后直接把predict复制过去    

**修改：**
1. predict.py-line122，线程数改为0
2. 初始参数除了他给的以外，建议还加上max_seq_length 280, bS 16
3. 修改了utils_wikisql.py-line1750，将'agg': 0, 'sel': 1修改成'agg': [0], 'sel': [1]（TableQA格式）
4. predict.py-line103，他原本只生成了sql语句，并没有执行，我这里再把执行结果写进去，虽然sel-col只有一列，但是我还是像agg、sel字段那样，放到了list里
5. 修改了dbengine.py-line117，由于我们要执行sql语句，他原本没有写直接执行的(为了安全考虑还是不再写了)，而是根据conds执行的，我们把原本处理英文的内容注释掉，免得它影响结果
6. 修改了predict.py多部分，new了一个compare_sql.py文件，专门用来比较两个sql（不是sql语句）是否相等，然后在predict中调用他（注意比较的时候忽略cond_conn_op）还有特别重要一点是：conds并不是按顺序排的，可能多个where条件顺序是不一样的，判断的时候要按照where-col排序
7. predict.py-line121，增加了一条输出语句，每100个bS也就是1600条输出一次
8. 修改了utils_wikisql.py-line1756/1764，jsonl->json
9. sqlova_ch.db和com_message.db是同一个数据库
10. 增加了一个agrs：have_sql，默认为False，如果为False就说明开始不给答案（类似TableQA的test），True的话就会给出准确率，也会给出答案

**更新：**
服务器的train.py、wikisql_models.py、utils.py、utils_wikisql.py、dbengine.py

### **\*\*\*\*\*\*\*\*\*\* 2020/12/04 \*\*\*\*\*\*\*\*\*\***
将财报表拆分成三个小表，并将格式改为TableQA的格式；观察服务器内存情况（发现内存溢出）
试着拿报表数据做训练或fine-tune
**拆分中**：
1. 放Table_financial_statements表到train.db/dev.db/test.db中，分别250w、150w、100w条		--db
2. 把Table_financial_statements表转成一行放到*.tables.json中					--表
3. 问题（ing...）  

**内存分析**：
在服务器中新建一个复制主文件的，进行内存分析


### **\*\*\*\*\*\*\*\*\*\* 2020/12/05 \*\*\*\*\*\*\*\*\*\***

 - 写在无sql语句的情况下做infer；将问题代入train/dev/test.json中；查看服务器train函数为什么用了40gb内存；试着用colab跑
 - 完成了未知答案的predict(只修改了predict)；
 - 发现了一个问题：wikisql_models.py-line28初始化max_wn=4，这样同时指定了每张表最多四列，如果有张表只有3列就会报错，除了服务器，其他地方都修改成了3

### **\*\*\*\*\*\*\*\*\*\* 2020/12/06 \*\*\*\*\*\*\*\*\*\***

 - 将financial_statements的单独问题合并到train/dev中，修改了test的格式(只剩question和tableid)；
 - 测试了financial_statements的单独问题

### **\*\*\*\*\*\*\*\*\*\* 2020/12/08 \*\*\*\*\*\*\*\*\*\***

 - 在原6次迭代基础上跑两次迭代，TableQA+fin基础上训练，最终准确率86%，继续跑迭代
 - 修改了predict里每20个bs输出的语句，（iB + 1） * args.bS
 - 用上面训练出来的参数预测单个fin，dev准确率：1613  /  1616
 - 修改了have_sql参数问题，以后不加--have_sql证明为False，加则为True[（原因）](https://blog.csdn.net/WILDCHAP_/article/details/110878484)

### **\*\*\*\*\*\*\*\*\*\* 2020/12/09 \*\*\*\*\*\*\*\*\*\***
 - 跑出来的6+2+2，准确率87.4%
 - 继续完成昨天的bS做无sql的predict代码完成
 - 修改utils_wikisql.py-line1764将一次性读取w换为附加读取'a+'

> **发现问题：**   
> 用stanza分词的和没用的结果差别巨大，于是用train的infer看一下是否只是predict_nosql的问题，发现还是查错，证明是stanza分词的问题
> 
> **解决方法：**   
> 先尝试修改predict里的predict_nosql方法（参照predict方法修改），然后再修改train里面的infer方法

### **\*\*\*\*\*\*\*\*\*\* 2020/12/11 \*\*\*\*\*\*\*\*\*\***

修改好了predict里的predict_nosql方法，达到了和predict方法一样的效果，再修改train的infer方法

### **\*\*\*\*\*\*\*\*\*\* 2020/12/12 \*\*\*\*\*\*\*\*\*\***

 - 创建了excel数据库，整理了表名，问题等信息
> **发现问题：** 
> 在加载real型问题时出错：找不到where-value起始位置  
> **解决方法：**   
> 修改annotate_ch_corenlp.py，在find时判断：如果是对real类型列进行操作就将它转换成str进行find

### **\*\*\*\*\*\*\*\*\*\* 2020/12/13 \*\*\*\*\*\*\*\*\*\***
创建了一个修改model专用的版本，按照train.py的执行顺序一个一个步骤修改  

 - 主要是三个问题：

**select-column个数，where-operate，where-value**
先修改前面两个问题   

### **\*\*\*\*\*\*\*\*\*\* 2020/12/14 \*\*\*\*\*\*\*\*\*\***
写出类WCCO类和其前馈方法，并在utils_wikisql.py-line1042写将权重矩阵转换成预测where之间连接词并返回的函数(没有足够的where连接则返回空list，如果返回下标是0[' ']则改为1['and'])写完之后发现好像用不到...增加了一个s_wcco输出

**修改损失计算函数：**
wikisql_models.pyy-line927添加了wcco的损失计算，并在line-1014写了计算wcco损失值的方法

**修改了获取标记数据：**
在train.py-line285修改了utils_wikisql.py的get_g方法，添加了对正确的wcco的获取（也是为了上一步计算损失值）

**发现问题：**
在TableQA中，'cond_conn_op'字段只有一个值，但当遇到三个where条件时，无法判定，这是TableQA的格式，我们认定他全为一种

### **\*\*\*\*\*\*\*\*\*\* 2020/12/15 \*\*\*\*\*\*\*\*\*\***
**修改train & test**
 - wcco取argmax时注意dtype(int64->float32)，修改了WOOC类，将其填充操作那里改成1维，最后对它进行降维（[16,3,3]==> [16,3]）
 - 修改train.py-line344根据s_*权重预测出真实可能的列的函数（utils_wikisql.py-line1188，还有之前写的pred_wcco函数），增加了对wcco的预测
 - 修改train.py-line352，生成对应的sql语句表示时，增加wcco的表示（utils_wikisql.py-line1770）
 - 修改train.py-line356，增加对wcco准确率的计算（utils_wikisql.py-line1684），这里单个元素判定函数可以用sc的判定函数get_cnt_sc_list，但下面的完美整体判断中也要加wcco
 - 修改train.py-line368，这里只要修改get_cnt_x_list函数的内容，这个函数调用了数据库的执行方法，所以必须修改数据库的执行方法，添加and和or连接条件
**下面修改数据库方法：**
 - 文件位置于debengine.py，为execute方法增加参数wcco，然后根据wcco判断用and还是用or连接，测试后效果可以，然后按照同样方法修改另一个execute函数
**继续修改train**
 - 修改train.py-line738，增加了每个epoch对wcco准确率的输出
 - 同样的方法修改test方法，注意在test方法里，我注释掉了往results列表里插值的循环和cnt_list里丢数据的方法(用处不大还占内存)
 - 注释了大量冗余代码，见'2020/12/15注释'

*至此，do_train修改完毕*

**修改infer：**

 - 修改train.py-line701_703，增加接收和使用wcco的s_wcco和pr_wcco
 - 修改train.py-line710，增加pr_wcco参数
 - 修改train.py-line721，执行数据库增加wcco

*至此，infer修改完毕*

**修改get_data：**
12/13修改，取消对wcco和sn的过滤
*至此，train.py修改完毕*

### **\*\*\*\*\*\*\*\*\*\* 2020/12/16 \*\*\*\*\*\*\*\*\*\***
**修改predict**

 - 修改line70，83，85，90增加一个g_wcco
 - 修改line120，增加数据库调用参数
 - 修改准确率判断函数，增加了“如果wcco不同，则也返回错误”的判断
 - 同样方法修改predict_nosql函数
 - 这里还要修改pr_sql_i ->
   pr_sql_q函数generate_sql_q中调用的generate_sql_q1方法(utils_wikisql.py-line1887)
   
*至此，predict.py修改完毕*

**SC：**

 - 注意sc的类要和sa一起改/写
 - 在utils_wikisql.py-line333中增加了wenc_hs大小[16, 17, 100] -> [16, 3, 17,
   100]，3代表sel的大小（1代表选中的第一列，2代表选中额第二列）
 - 修改utils_wikisql.py-line389将sel和agg的返回值g_*返回成二维list类型
 - 还有许多细节如0维度的负无穷填充见文件注释

**SN（select-number）：**

 - 修改了wikisql_models.py写了一个SN类，并在utils_wikisql.py中写了一个pred_sn方法-由sn得分预测sn
 - sn是为了确定sn数量，当然，这里也可以通过高维sc/sa实现

### **\*\*\*\*\*\*\*\*\*\* 2020/12/17 \*\*\*\*\*\*\*\*\*\***
在上面pred_sn方法里，有个错误要注意：在模型迭代次数还没上去的时候，有可能出现选中两个同一列的情况
**SA：**

 - 修改完毕注意expand维度，还要修改pred_sa方法(虽然只在pre时会调用它)
 - 更多见代码注释

在train上面的get_g方法中加g_sn返回，model返回也加s_sn，最后再把sn加入损失计算中，这里还要修改sc和sa的损失函数（之前是一维）

> 发现调优方法：既然我们已经有了s_sn，就不需要sc为[16,2,14]维了，只要[16,14]就行，所以把昨天写的SC改回二维...

sn要注意，维度的取值是0/1，真实值只能取1/2，所以在计算损失值的时候要用新的计算
修改了sn的维度[16,2] -> [16,3]，不然损失函数会报取值错误，注意也要修改sa的维度[16,2,6] -> [16,3,6]
发现了：model调用参数里少了g_wo，于是增加了g_wo的调用

### **\*\*\*\*\*\*\*\*\*\* 2020/12/18 \*\*\*\*\*\*\*\*\*\***
整理wcco、sn、sc、sa结构

 - 修改通过sc等得分预测出sc等选择值的函数pred_sw_se
 - 修改通过预测出来的列生成伪sql语句的函数generate_sql_i，sel和agg不需要外面list括起来了
 - 修改计算伪sql各单元准确率的get_cnt_sw_list函数，增加了sn的计算（sc和sa并不需要修改）
 - 修改通过伪sql各单元连接数据库查询执行的数据库查询方法，增加了对多列sn和sa的查询

> 发现问题：
> 1.初始化db时加self.db = self.db.get_connection()，sql语句去除AS result
> 2.在遇到where col_2 = col_2 or col_2 = col_2的语句时会取到同一个val，所以这里改成col_2=0 or col_2 = 1的形式

同样修改输出准确率的函数，输出sn的准确率

### **\*\*\*\*\*\*\*\*\*\* 2020/12/19 \*\*\*\*\*\*\*\*\*\***
*train函数已修改完毕，继续修改test函数*
*test函数已修改完毕，继续修改infer函数*
*train.py修改完毕，修改predict.py*
修改生成sql语句的utils_wikisql.py-line1951的generate_sql_q1函数
具体见代码注释

> 发现问题：WV大写会自动转换成小写
> 最后发现是predict.py-line88的generate_sql_i函数出了问题，这涉及到所有文件，根源问题出现在utils_wikisql.py的-line1182的merge_wv_t1_eng方法上，把utils_wikisql.py的line-1187的nlq= NLq.lower()改为nlq = NLq，还有所有的.lower()都删掉
> 
修改完predict.py函数后，修改结果判断函数
至此，where之间连接条件 and & or 和select列个数已经实现，再写一个轻量训练的版本
train：不连接数据库获取结果,比较结果，最后不反回acc_x
test： 不连接数据库获取结果,比较结果，最后不反回acc_x