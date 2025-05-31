from pymysql import *
import scipy.stats as stats

# 十几个无数据，均无关注，求相关时数据中删除
# 是 1 否 0

def main():
    xh = []
    nl = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh,csrq from v_bzks_atj where nj = "2018"')
    print("查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(jg[0])
        try:
            nl.append(2018 - int(str(jg[1]).split("-")[0]))
        except:
            print(jg[0][:2])
            if int(jg[0][:2]) == 16:
                nl.append(2018-1998)
            if int(jg[0][:2]) == 17:
                nl.append(2018-1999)
        gz.append(0)
    print(len(xh),len(nl),len(gz))


    count2 = cs1.execute('select xh from v_gzmd_atj')
    print("查询到%d条数据:" % count2)

    for i in range(count2):
        # 获取查询的结果
        jg = cs1.fetchone()
        if jg[0] in xh:
            gz[xh.index(jg[0])] = 1

    # 关闭Cursor对象
    cs1.close()
    conn.close()

    f = open("./年龄.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(nl[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, nl)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()