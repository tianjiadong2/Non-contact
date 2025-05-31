from pymysql import *
import scipy.stats as stats
import numpy as np

# 社交
# 关注名单 1 否 0

def main():
    xh = []
    dd = []
    jyrq = []
    jysj = []
    shejiao_dict = {}

    conn = connect(host='')
    cs1 = conn.cursor()
    # 处理消费次数和消费总金额
    count = cs1.execute("SELECT xh,dd,jyrq,jysj FROM `v_xfjl_atj` where jyrq >'2019-01-10' and jyrq <='2019-02-10'  ORDER BY jyrq,jysj")
    print("消费记录查询到%d条数据:" % count)
    # 存储数据
    for i in range(count):
        jg = cs1.fetchone()
        xh.append(str(jg[0]).strip())
        dd.append(str(jg[1]).strip())
        jyrq.append(str(jg[2]).strip())
        jysj.append(str(jg[3]).strip())
    # 处理数据
    for i in range(len(xh)):     # 大循环
        if i % 1000 == 0:        # 计算过程标记
            print(i)
        for j in range(i+1,min(i+21,len(xh))):         # 小循环，先计算两个时间
            time_i = (60 * int(str(jysj[i]).strip().split(":")[0]) + int(str(jysj[i]).strip().split(":")[1]))
            time_j = (60 * int(str(jysj[j]).strip().split(":")[0]) + int(str(jysj[j]).strip().split(":")[1]))
            if abs(time_j-time_i)<2 and str(dd[i]).strip()==str(dd[j]).strip():   # 时间差为0或1，且地点相同
                xh_key = ""
                if str(xh[i]).strip()<str(xh[j]).strip():             # 统一key，小号在前，并去除自己
                    xh_key = str(xh[i]+":"+xh[j])
                elif str(xh[i]).strip()>str(xh[j]).strip():
                    xh_key = str(xh[j]+":"+xh[i])

                if xh_key not in shejiao_dict:
                    shejiao_dict[xh_key] = 1
                else:
                    shejiao_dict[xh_key] = shejiao_dict[xh_key] + 1
            elif abs(time_j-time_i) > 1:
                break

    # # 删除小于 3 的偶现
    # for key in list(shejiao_dict.keys()):
    #     if shejiao_dict[key] < 5:
    #         del shejiao_dict[key]
    #     else:
    #         print(key,shejiao_dict[key])
    # print(len(shejiao_dict))

    np.save('./shejiao_dict_11.npy', shejiao_dict)  # 持久化


if __name__ == '__main__':
    main()