from pymysql import *
import scipy.stats as stats

# 超市消费总额2018年
# 关注名单 1 否 0

def main():
    xh = []
    xfze = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
    print("学号查询到%d条数据:" % count)

    # 处理学号和初始化其他
    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(jg[0])
        xfze.append(0)
        gz.append(0)

    # 处理消费次数和消费总金额
    count2 = cs1.execute("SELECT * FROM `v_xfjl_atj` where jyrq >'2019-01-10' and jyrq <'2020-01-10'")
    print("消费记录查询到%d条数据:" % count2)
    for i in range(count2):
        # 获取查询的结果
        if i % 10000 == 0:
            print(i)
        jg = cs1.fetchone()

        if jg[0] in xh:
            xfze[xh.index(jg[0])] = xfze[xh.index(jg[0])] + float(jg[4])

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

    f = open("./总消费总额.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(xfze[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, xfze)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()