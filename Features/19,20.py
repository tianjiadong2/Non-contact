from pymysql import *
import scipy.stats as stats

# 借阅图书，次数和累计分数
# 关注名单 1 否 0

def main():
    xh = []
    jycs = []
    jylj = []
    gz = []
    shuji = {}

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
    print("学号查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        xh.append(str(jg[0]).strip())
        jycs.append(0)
        jylj.append(0)
        gz.append(0)

    # 书籍分数
    for line in open("./书名.csv", "r"):
        try:
            shuji[line.rsplit(",",1)[0]] = int(line.rsplit(",",1)[1].strip())
        except:
            print(line)
    print(shuji)

    # 借阅记录处理
    count2 = cs1.execute("SELECT xh,sm FROM v_ts_atj group by xh,sm")
    print("图书查询到%d条数据:" % count2)
    for i in range(count2):
        if i % 10000 == 0:
            print(i)
        jg = cs1.fetchone()
        if str(jg[0]).strip() in xh:
            jycs[xh.index(jg[0])] = jycs[xh.index(jg[0])] + 1
            if jg[1] in shuji:
                jylj[xh.index(jg[0])] = jylj[xh.index(jg[0])] + shuji[jg[1]]

    count3 = cs1.execute('select xh,jbdm from v_gzmd_atj')
    print("关注名单查询到%d条数据:" % count3)
    for i in range(count3):
        # 获取查询的结果
        jg = cs1.fetchone()
        if str(jg[0]).strip() in xh:
            gz[xh.index(jg[0])] = jg[1]

    # 关闭Cursor对象
    cs1.close()
    conn.close()

    f = open("./借阅次数+计分.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(jycs[i])+","+str(jylj[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, jycs)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, jylj)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()