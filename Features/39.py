from pymysql import *
import scipy.stats as stats

# 汉 0  少数民族 1  缺省 删除
# 是 1 否 0

def main():
    xh = []
    mz = []
    gz = []

    conn = connect(host='')
    cs1 = conn.cursor()
    count = cs1.execute('select xh,mz from v_bzks_atj where nj = "2018"')
    print("查询到%d条数据:" % count)

    for i in range(count):
        # 获取查询的结果
        jg = cs1.fetchone()
        if jg[1] == "汉族":
            xh.append(jg[0])
            mz.append(0)
        elif jg[1] != "汉族" and jg[1] != NULL:
            xh.append(jg[0])
            mz.append(1)
        else:
            continue
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

    f = open("./民族.csv", "w")
    for i in range(len(xh)):
        f.write(str(xh[i])+","+str(mz[i])+","+str(gz[i])+"\n")
    f.close()

    # 求相关性
    coef, pvalue = stats.pointbiserialr(gz, mz)
    print('pointbiserialr', coef)
    print('pvalue', pvalue)

if __name__ == '__main__':
    main()