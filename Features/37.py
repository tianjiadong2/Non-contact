from pymysql import *
import scipy.stats as stats

# 男 1 女 0
# 是 1 否 0

def main():
    xh = []
    xb = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh,xb from v_bzks_atj where nj = "2018"')
    print("查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(jg[0])
        if jg[1] == "男":
            xb.append(1)
        else:
            xb.append(0)
        gz.append(0)

    count2 = cs1.execute('select xh, jbdm from v_gzmd_atj')
    print("查询到%d条数据:" % count2)

    for i in range(count2):
        # 获取查询的结果
        jg = cs1.fetchone()
        if jg[0] in xh:
            gz[xh.index(jg[0])] = jg[1]

    # 关闭Cursor对象
    cs1.close()
    conn.close()

    f = open("./性别.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(xb[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, xb)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

    r = stats.pearsonr(gz, xb)
    print("pearson系数：", r[0])
    print("P-Value：", r[1])

if __name__ == '__main__':
    main()