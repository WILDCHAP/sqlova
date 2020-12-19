# From original SQLNet code.
# Wonseok modified. 20180607

import records
import re
from babel.numbers import parse_decimal, NumberFormatError
from sqlnet.utils_col import degrad_c


schema_re = re.compile(r'\((.+)\)') # group (.......) dfdf (.... )group
num_re = re.compile(r'[-+]?\d*\.\d+|\d+') # ? zero or one time appear of preceding character, * zero or several time appear of preceding character.
# Catch something like -34.34, .4543,
# | is 'or'

# agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
# cond_ops = ['=', '>', '<', 'OP']
agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
cond_ops = ['>', '<', '=', '<>']

class DBEngine:

    def __init__(self, fdb):
        #fdb = 'data/test.db'
        # z1 = 'sqlite:///{}'.format(fdb)+'?check_same_thread=False'
        # print(z1)
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.db = self.db.get_connection()
        # #self.db = records.Database('sqlite:///?check_same_thread=False')
        # self.db = records.Database(z1)
        #self.fdb = fdb
        #print("1")

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('Table'):
            table_id = 'Table_{}'.format(table_id.replace('-', '_'))
        # add
        # conn = records.Database('sqlite:///{}'.format(self.fdb)).get_connection()
        # add
        #table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql.replace('\n','')

        #qu = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id)
        #qu = self.db.query("SELECT sql from sqlite_master WHERE tbl_name = 'Table_43ad6bdc1d7111e988a6f40f24344a08'")
        #qu = self.db.query("SELECT sql from sqlite_master WHERE tbl_name = 'table_f65ceb4f453d11e9b16ef40f24344a08'")
        qu = self.db.query("SELECT sql from sqlite_master WHERE tbl_name = :name", name=table_id)
        qu_2 = qu.all()
        qu_3 = qu_2[0]
        table_info = qu_3.sql.replace('\n','')

        #table_info = conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql.replace('\n', '')
        schema_str = schema_re.findall(table_info)[0]   #获取表中每个字段的名字和其对应的类型
        schema = {}
        for tup in schema_str.split(','):
            c, t = tup.split()
            #这里对c进行处理，col_1=>col_0
            #c = degrad_c(c)
            schema[c] = t
        select = 'col_{}'.format(select_index + 1)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            # if lower and (isinstance(val, str) or isinstance(val, str)):
            #     val = val.lower()
            #if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
            if schema['col_{}'.format(col_index + 1)] == 'real' and not isinstance(val, (int, float)):  #如果数据列是real类型，且wv不是数字类型
                try:
                    # print('!!!!!!value of val is: ', val, 'type is: ', type(val))
                    # val = float(parse_decimal(val)) # somehow it generates error.
                    val = float(parse_decimal(val, locale='en_US'))
                    # print('!!!!!!After: val', val)

                except NumberFormatError as e:
                    try:
                        val = float(num_re.findall(val)[0]) # need to understand and debug this part.
                    except:
                        # Although column is of number, selected one is not number. Do nothing in this case.
                        pass
            where_clause.append('col_{} {} :col_{}'.format(col_index + 1, cond_ops[op], col_index + 1))
            where_map['col_{}'.format(col_index + 1)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        #print query
        out = self.db.query(query, **where_map)
        #out = conn.query(query, **where_map)


        return [o.result for o in out]
    def execute_return_query(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('Table'):
            '''2020/12/01修改：table_{}-->Table_{}'''
            table_id = 'Table_{}'.format(table_id.replace('-', '_'))
        #table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql.replace('\n','')
        '''2020/12/01修改：类似execute修改函数，一句拆成四句'''
        qu = self.db.query("SELECT sql from sqlite_master WHERE tbl_name = :name", name=table_id)
        qu_2 = qu.all()
        qu_3 = qu_2[0]
        table_info = qu_3.sql.replace('\n', '')

        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(','):
            c, t = tup.split()
            schema[c] = t
        select = 'col_{}'.format(select_index + 1)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            # if lower and (isinstance(val, str) or isinstance(val, str)):
            #     val = val.lower()
            if schema['col_{}'.format(col_index + 1)] == 'real' and not isinstance(val, (int, float)):
                try:
                    # print('!!!!!!value of val is: ', val, 'type is: ', type(val))
                    # val = float(parse_decimal(val)) # somehow it generates error.
                    val = float(parse_decimal(val, locale='en_US'))
                    # print('!!!!!!After: val', val)

                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col_{} {} :col_{}'.format(col_index + 1, cond_ops[op], col_index + 1))
            where_map['col_{}'.format(col_index + 1)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        #print query
        out = self.db.query(query, **where_map)


        return [o.result for o in out], query
    def show_table(self, table_id):
        if not table_id.startswith('Table'):
            '''2020/12/01修改：table_{}-->Table_{}'''
            table_id = 'Table_{}'.format(table_id.replace('-', '_'))
        rows = self.db.query('select * from ' +table_id)
        print(rows.dataset)