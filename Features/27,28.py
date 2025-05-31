from pymysql import *
import scipy.stats as stats
import numpy as np

# 异常在校天数，离校天数
# 关注名单 1 否 0

def main():
    xh = []
    zaixiao = []
    lixiao = []
    gz = []
    riqi_dict = {}

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
    print("学号查询到%d条数据:" % count)

    # 处理学号和初始化其他
    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(jg[0])
        zaixiao.append(0)
        lixiao.append(0)
        gz.append(0)

    # 处理全日期字典
    f4 = open('./日期特征.csv', encoding='utf-8')
    lines = f4.readlines()
    for line in lines:
        riqi_dict[line.split(',')[0]] = int(line.split(',')[1])
    f4.close()

    # 挨个处理学号
    # 查询**时间内消费记录，构建全日期字典，学生日期字典，遍历全日期字典，判断是否在学生字典，是否节假日，四种情况讨论
    for i in xh:

        count2 = cs1.execute("SELECT jyrq FROM v_xfjl_atj WHERE xh='" + str(i) + "' and jyrq >'2019-01-10' and jyrq <'2020-01-10' GROUP BY jyrq")
        xuesheng_riqi = []
        for ii in range(count2):
            jg = cs1.fetchone()
            xuesheng_riqi.append(str(jg[0]))
        for key in riqi_dict:
            if key in xuesheng_riqi and riqi_dict[key] == 0:
                zaixiao[xh.index(i)] = zaixiao[xh.index(i)] + 1
            elif key not in xuesheng_riqi and riqi_dict[key] == 1:
                lixiao[xh.index(i)] = lixiao[xh.index(i)] + 1
        print(xh.index(i), i, zaixiao[xh.index(i)], lixiao[xh.index(i)])

    count3 = cs1.execute('select xh from v_gzmd_atj')
    print("关注名单查询到%d条数据:" % count3)
    for i in range(count3):
        # 获取查询的结果
        jg = cs1.fetchone()
        if jg[0] in xh:
            gz[xh.index(jg[0])] = 1

    # 关闭Cursor对象
    cs1.close()
    conn.close()

    f = open("./异常在校离校.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(zaixiao[i])+","+str(lixiao[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, zaixiao)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, lixiao)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()