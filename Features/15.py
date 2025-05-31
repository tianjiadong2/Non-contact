from pymysql import *
import scipy.stats as stats

# 挂科科目数
# 关注名单 1 否 0

def main():
    xh = []
    gkkm = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
    print("学号查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(str(jg[0]).strip())
        gkkm.append(0)
        gz.append(0)

    count2 = cs1.execute("SELECT xh,zpcj FROM `v_cj_atj`")
    print("成绩查询到%d条数据:" % count2)
    for i in range(count2):
        # 获取查询的结果
        if i % 10000 == 0:
            print(i)

        jg = cs1.fetchone()
        if str(jg[0]).strip() in xh:
            if jg[1] is None:
                gkkm[xh.index(jg[0])] = gkkm[xh.index(jg[0])] + 1
            elif float(jg[1]) < 60:
                gkkm[xh.index(jg[0])] = gkkm[xh.index(jg[0])] + 1

    count3 = cs1.execute('select xh from v_gzmd_atj')
    print("关注名单查询到%d条数据:" % count3)
    for i in range(count3):
        # 获取查询的结果
        jg = cs1.fetchone()
        if str(jg[0]).strip() in xh:
            gz[xh.index(jg[0])] = 1

    # 关闭Cursor对象
    cs1.close()
    conn.close()

    f = open("./挂科科目数.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(gkkm[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, gkkm)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()