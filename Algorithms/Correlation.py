from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from pymysql import *
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler

df_wine = pd.read_csv('G:/博士_组会/算法/数据整合_17_相关性.csv',header=None)
df_wine.columns = ['学号','周末假期在校天数', '异性交流总频次','平均每餐交流次数1', '紊乱度1','正常时间离校天数','室友交流总频次', '平均每餐交流次数2',
                   '紊乱度2', '电控缴费次数','三餐平均消费','超市消费总额', '总就餐次数','其他消费次数','总消费金额','年社交频次','平均每餐1',
                   '紊乱度3','留级','学分积','挂科科目数','奖学金次数','奖学金金额','团体奖学金次数','借阅次数','借阅书籍与心理相关','违纪次数'
                    ,'籍贯','同学交流总频次','平均每餐2','紊乱度4','性别','年龄','民族','助学金次数','助学金金额','就医次数','早餐时间方差',
                   '午餐时间方差','晚餐时间方差','失眠','类别']
df_wine['类别'].value_counts()

y = df_wine['类别'].values
X = df_wine[['周末假期在校天数', '异性交流总频次','平均每餐交流次数1', '紊乱度1','正常时间离校天数','室友交流总频次', '平均每餐交流次数2',
                   '紊乱度2', '电控缴费次数','三餐平均消费','超市消费总额', '总就餐次数','其他消费次数','总消费金额','年社交频次','平均每餐1',
                   '紊乱度3','留级','学分积','挂科科目数','奖学金次数','奖学金金额','团体奖学金次数','借阅次数','借阅书籍与心理相关','违纪次数'
                    ,'籍贯','同学交流总频次','平均每餐2','紊乱度4','性别','年龄','民族','助学金次数','助学金金额','就医次数','早餐时间方差',
                   '午餐时间方差','晚餐时间方差']].values

# zm = df_wine['平均每餐交流次数'].values.reshape(-1, 1)
#
# robust_scaler = RobustScaler()
# zm_robust = robust_scaler.fit_transform(zm)
# print(len(y),len(zm),len(zm_robust))
# print(y)
# print(zm)
#
# coef, pvalue = stats.pointbiserialr(y, zm_robust)
# print('pointbiserialr', coef)
# print('pvalue', pvalue)

for i in df_wine.columns:
    zm = df_wine[i].values.reshape(-1,1)

    robust_scaler = RobustScaler()
    zm_robust = robust_scaler.fit_transform(zm)

    coef, pvalue = stats.pointbiserialr(y, zm_robust)
    print(i+","+str(coef)+","+str(pvalue))
    #print('pointbiserialr', coef)
    #print('pvalue', pvalue)