"""
@Time ： 2020/12/3 17:45
@Auth ： WILDCHAP
@File ：compare_sql.py
@IDE ：PyCharm
"""
# def compare(sql, pr_sql, ans, pr_ans):
#     right_ans = 0
#     # 比较长度
#     if len(sql) != len(pr_sql):
#         return 0
#     # 一个一个比
#     for index, (val, pr_val) in enumerate(zip(sql, pr_sql)):
#         # 比较agg
#         if val['agg'] != pr_val['agg']:
#             continue
#         # 比较sel
#         if val['sel'] != pr_val['sel']:
#             continue
#         # 比较conds
#         if compare_conds(val['conds'], pr_val['conds']) == False:
#             continue
#
#         right_ans += 1
#
#     return True

def compare(val, pr_val, ans, pr_ans):
    # 比较agg
    if val['agg'] != pr_val['agg']:
        return False
    # 比较conds
    if compare_conds(val['conds'], pr_val['conds']) == False:
        return False
    # 比较sel，不同如果是count的话，再比较结果
    if val['sel'] != pr_val['sel']:
        if val['agg'] != [4]:
            return False
        return compare_by_result(ans, pr_ans)
    return True

# 一样返回True
def compare_conds(conds, pr_conds):
    # 比较长度
    if len(conds) != len(pr_conds):
        return False
    # 将conds都按照where-col排序
    conds.sort(key=lambda u:(u[0]))
    pr_conds.sort(key=lambda u: (u[0]))
    # 一项一项比
    for index, (conds_val, pr_conds_val) in enumerate(zip(conds, pr_conds)):
        if conds_val != pr_conds_val:
            return False

    return True

# 传入一条结果和一条结果比
def compare_by_result(result, pr_result):
    result.sort()
    pr_result.sort()
    if result==pr_result:
        return True
    return False