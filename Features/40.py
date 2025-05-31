from pymysql import *
import scipy.stats as stats
from parseIdCard import parseIdCard

# 籍贯， 深圳 1， 广东 2， 其他省份 3，国外 5
# 是 1 否 0

def main():
    xh = []
    jiguan = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh,sfzjh from v_bzks_atj where nj = "2018"')
    print("查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        idcardinfo = parseIdCard.parseArea(jg[1])
        if idcardinfo['area'][0:4] == "广东深圳":
            xh.append(jg[0])
            jiguan.append(1)
        elif idcardinfo['area'][0:2] == "广东":
            xh.append(jg[0])
            jiguan.append(2)
        elif idcardinfo['area'][0:2] == "非法":
            xh.append(jg[0])
            jiguan.append(5)
        else:
            xh.append(jg[0])
            jiguan.append(3)
        gz.append(0)

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

    f = open("./籍贯.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(jiguan[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, jiguan)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()