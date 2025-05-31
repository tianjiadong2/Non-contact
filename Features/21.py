from pymysql import *
import scipy.stats as stats

# 违纪次数 int group by去重
# 关注名单 1 否 0

def main():
    xh = []
    wjcs = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
    print("学号查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(jg[0])
        wjcs.append(0)
        gz.append(0)

    count2 = cs1.execute("SELECT xh FROM `v_atj_wjcf`")
    print("图书查询到%d条无重复数据:" % count2)
    for i in range(count2):
        # 获取查询的结果
        jg = cs1.fetchone()
        if jg[0] in xh:
            wjcs[xh.index(jg[0])] = wjcs[xh.index(jg[0])] + 1

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

    f = open("./违纪次数.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(wjcs[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, wjcs)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()