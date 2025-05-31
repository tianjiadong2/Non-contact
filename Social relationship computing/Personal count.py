import numpy as np
from pymysql import *
import scipy.stats as stats

xh = []
sjcs = {}
gz = []

shejiao_dict = np.load('./shejiao_dict_201901.npy',allow_pickle=True).item()

conn = connect(host='')
cs1 = conn.cursor()
count = cs1.execute('select xh from v_bzks_atj where nj = "2018"')
print("学号查询到%d条数据:" % count)

for i in range(count):
    # 获取查询的结果
    jg = cs1.fetchone()
    xh.append(str(jg[0]).strip())
    sjcs[str(jg[0]).strip()] = 0
    gz.append(0)

print("字典共查询到%d条数据:" % len(shejiao_dict))
i = 0
for key in shejiao_dict.keys():
    if i % 1000 == 0:
        print(i)
    i=i+1
    try:
        stu_1 = str(key).split(":")[0]
        stu_2 = str(key).split(":")[1]

        if stu_1 in sjcs:
            sjcs[stu_1] = sjcs[stu_1] + shejiao_dict[key]
        if stu_2 in sjcs:
            sjcs[stu_2] = sjcs[stu_2] + shejiao_dict[key]
    except:
        print(key)
        continue

# 关闭Cursor对象
cs1.close()
conn.close()
print(sjcs)

np.save('./sjcs_201807.npy', sjcs)  # 持久化