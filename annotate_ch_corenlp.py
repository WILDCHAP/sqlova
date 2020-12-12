#from train import get_data

'''我们不修改原数据，只在从硬盘读入的时候进行查找（待优化）'''
'''根据数据已有值查出索引【起始，结束】'''
def get_star_and_end(t):
    #WN数，也是WV数
    wv_n = len(t['sql']['conds'])
    wv_List = []
    #用一个for循环遍历出一个数列
    for inx in range(wv_n):
        wv_temp = []
        wv_val = t['sql']['conds'][inx][2] #获取第inx个value
        '''2020/12/12修改：如果是对real类型列进行操作就将它转换成str进行find'''
        if not isinstance(wv_val, str):
            wv_val = str(wv_val)
        beg = t['question'].find(wv_val)
        end = beg + len(wv_val) - 1
        if(beg == -1 or end == -1):
            return -1
        wv_temp.append(beg)
        wv_temp.append(end)
        wv_List.append(wv_temp)
    return wv_List
