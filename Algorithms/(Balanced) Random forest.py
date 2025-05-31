from datetime import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import balanced_accuracy_score,plot_confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

df_wine = pd.read_csv('G:/博士_组会/算法/数据整合_17.csv',header=None)
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
                   '午餐时间方差','晚餐时间方差','失眠']].values
#

# score_lt = []
#
# for i in range(1,40,1):
#
#     rfc = RandomForestClassifier(n_estimators=257,max_features=8,max_depth=6)
#     y_pred = cross_val_predict(rfc, X, y, cv=10)
#     score_ = balanced_accuracy_score(y, y_pred)
#     score_lt.append(score_)
#     print(str(i)+":"+str(score_))
#
# score_max = max(score_lt)
# print('最大得分：{}'.format(score_max),
#       '子树数量为：{}'.format(score_lt.index(score_max)*10+1))
#
# # 绘制学习曲线
# x = np.arange(0,40,1)
# plt.subplot(111)
# plt.plot(x, score_lt, 'r-')
# plt.show()



rfc = RandomForestClassifier(n_estimators=100)
y_pred = cross_val_predict(rfc, X, y, cv=10)
y_pred_proba = cross_val_predict(rfc, X, y, cv=10,method='predict_proba')

print("EasyEnsemble classifier performance:")
print(
    f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.2f} - "
    f"Geometric mean {geometric_mean_score(y, y_pred):.2f}"
)

print(confusion_matrix(y, y_pred))


target_names = ['class 0', 'class 1']
print(classification_report_imbalanced(y, y_pred, target_names=target_names))

ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()
plt.show()

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X, y)
f, ax = plt.subplots(figsize=(7, 5))
for i in rf.feature_importances_:
    print(i)


ax.bar(range(len(rf.feature_importances_)),rf.feature_importances_)
ax.set_title("Feature Importances")

f.show()
plt.show()


# top k
y_pred_proba_ = []
for i in y_pred_proba:
    y_pred_proba_.append(i[1])

A = y_pred_proba_
B = y

Z = zip(A, B)
Z = sorted(Z, reverse=True)

A_new, B_new = zip(*Z)

for i in range(len(A_new)):
    if i in [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        print(str(i) + ":" + str(sum(B_new[0:i])))
