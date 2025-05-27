import numpy as np
from pymysql import *
import scipy.stats as stats

sjcs_201901 = np.load('./sjcs_201901_tongxue.npy',allow_pickle=True).item()
sjcs_201902 = np.load('./sjcs_201902_tongxue.npy',allow_pickle=True).item()
sjcs_201903 = np.load('./sjcs_201903_tongxue.npy',allow_pickle=True).item()
sjcs_201904 = np.load('./sjcs_201904_tongxue.npy',allow_pickle=True).item()
sjcs_201905 = np.load('./sjcs_201905_tongxue.npy',allow_pickle=True).item()
sjcs_201906 = np.load('./sjcs_201906_tongxue.npy',allow_pickle=True).item()
sjcs_201907 = np.load('./sjcs_201907_tongxue.npy',allow_pickle=True).item()
sjcs_201908 = np.load('./sjcs_201908_tongxue.npy',allow_pickle=True).item()
sjcs_201909 = np.load('./sjcs_201909_tongxue.npy',allow_pickle=True).item()
sjcs_201910 = np.load('./sjcs_201910_tongxue.npy',allow_pickle=True).item()
sjcs_201911 = np.load('./sjcs_201911_tongxue.npy',allow_pickle=True).item()
sjcs_201912 = np.load('./sjcs_201912_tongxue.npy',allow_pickle=True).item()


xh = []
sjcs_nian = {}
sj = []
xfzcs = []
pjcs = []
wld = []
gz = []

conn = connect(host='')
cs1 = conn.cursor()
count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
print("学号查询到%d条数据:" % count)

for i in range(count):
    # 获取查询的结果
    jg = cs1.fetchone()
    xh.append(str(jg[0]).strip())
    xfzcs.append(0)
    pjcs.append(0)
    wld.append(0)
    gz.append(0)

for i in xh:
    sjcs_nian[i] = sjcs_201901[i]+sjcs_201902[i]+sjcs_201903[i]+sjcs_201904[i]+sjcs_201905[i]+sjcs_201906[i]+sjcs_201907[i]+sjcs_201908[i]+sjcs_201909[i]+sjcs_201910[i]+sjcs_201911[i]+sjcs_201812[i]
    wld[xh.index(i)] = abs(sjcs_201912[i]-sjcs_201911[i]) + abs(sjcs_201911[i]-sjcs_201910[i]) + abs(sjcs_201910[i]-sjcs_201909[i])
             +abs(sjcs_201909[i]-sjcs_201908[i]) + abs(sjcs_201908[i]-sjcs_201907[i]) +abs(sjcs_201907[i]-sjcs_201906[i]
			 +abs(sjcs_201906[i]-sjcs_201905[i]) + abs(sjcs_201905[i]-sjcs_201904[i]) +abs(sjcs_201904[i]-sjcs_201903[i]
			 +abs(sjcs_201903[i]-sjcs_201902[i]) + abs(sjcs_201902[i]-sjcs_201901[i]))
np.save('./sjcs_nian_tongxue.npy', sjcs_nian)  # 持久化

for i in xh:
    sj.append(sjcs_nian[i])


# 处理消费次数
count2 = cs1.execute("SELECT xh,count(xh) FROM `v_xfjl_atj`  WHERE jyrq >'2019-01-10' AND jyrq <'2020-01-10' GROUP BY xh")
for i in range(count2):
    jg = cs1.fetchone()
    if jg[0] in xh:
        xfzcs[xh.index(jg[0])] = int(jg[1])

# 平均每次消费的社交次数
for i in xh:
    if xfzcs[xh.index(i)] !=0:
        pjcs[xh.index(i)] = sj[xh.index(i)]/xfzcs[xh.index(i)]


# 关注代码
count3 = cs1.execute('select xh,jbdm from v_gzmd_atj')
print("关注名单查询到%d条数据:" % count3)
for i in range(count3):
    # 获取查询的结果
    jg = cs1.fetchone()
    if jg[0] in xh:
        gz[xh.index(jg[0])] = jg[1]

# 关闭Cursor对象
cs1.close()
conn.close()

f = open("./同学_年社交次数+平均+紊乱度.csv", "w")
for i in range(len(xh)):
    f.write(str(xh[i])+","+str(sj[i])+","+str(xfzcs[i])+","+str(pjcs[i])+","+str(wld[i])+","+str(gz[i])+"\n")
f.close()

# 求相关性
coef, pvalue = stats.pointbiserialr(gz, sj)
print('pointbiserialr', coef)
print('pvalue', pvalue)

# 求相关性
coef, pvalue = stats.pointbiserialr(gz, xfzcs)
print('pointbiserialr', coef)
print('pvalue', pvalue)

# 求相关性
coef, pvalue = stats.pointbiserialr(gz, pjcs)
print('pointbiserialr', coef)
print('pvalue', pvalue)

# 求相关性
coef, pvalue = stats.pointbiserialr(gz, wld)
print('pointbiserialr', coef)
print('pvalue', pvalue)