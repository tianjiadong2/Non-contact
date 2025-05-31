# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

print(__doc__)

###############################################################################
# Data loading
###############################################################################

from collections import Counter

import numpy as np
import pandas as pd

###############################################################################
# First, you should download the Porto Seguro data set from Kaggle. See the
# link in the introduction.
#training_data = pd.read_csv("./train.csv")
#testing_data = pd.read_csv("./test.csv")

#y_train = training_data[["id", "target"]].set_index("id")
#X_train = training_data.drop(["target"], axis=1).set_index("id")
#X_test = testing_data.set_index("id")

df_wine = pd.read_csv('G:/博士_组会/算法/数据整合_17.csv',header=None)
df_wine.columns = ['学号','周末假期在校天数', '异性交流总频次','平均每餐交流次数', '紊乱度1','正常时间离校天数','室友交流总频次', '平均每餐交流次数',
                   '紊乱度2', '电控缴费次数','三餐平均消费','超市消费总额', '总就餐次数','其他消费次数','总消费金额','年社交频次','平均每餐',
                   '紊乱度3','留级','学分积','挂科科目数','奖学金次数','奖学金金额','团体奖学金次数','借阅次数','借阅书籍与心理相关','违纪次数'
                    ,'籍贯','同学交流总频次','平均每餐','紊乱度4','性别','年龄','民族','助学金次数','助学金金额','就医次数','早餐时间方差',
                   '午餐时间方差','晚餐时间方差','失眠','类别']
df_wine['类别'].value_counts()

y_train = df_wine['类别'].values
X_train = df_wine[['周末假期在校天数', '异性交流总频次','平均每餐交流次数', '紊乱度1','正常时间离校天数','室友交流总频次', '平均每餐交流次数',
                   '紊乱度2', '电控缴费次数','三餐平均消费','超市消费总额', '总就餐次数','其他消费次数','总消费金额','年社交频次','平均每餐',
                   '紊乱度3','留级','学分积','挂科科目数','奖学金次数','奖学金金额','团体奖学金次数','借阅次数','借阅书籍与心理相关','违纪次数'
                    ,'籍贯','同学交流总频次','平均每餐','紊乱度4','性别','年龄','民族','助学金次数','助学金金额','就医次数','早餐时间方差',
                   '午餐时间方差','晚餐时间方差']].values

y_train=pd.DataFrame(y_train)
X_train=pd.DataFrame(X_train)
###############################################################################
# The data set is imbalanced and it will have an effect on the fitting.

print(f"The data set is imbalanced: {Counter(y_train)}")
print(df_wine['类别'].value_counts())


###############################################################################
# Define the pre-processing pipeline
###############################################################################

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def convert_float64(X):
    return X.astype(np.float64)


###############################################################################
# We want to standard scale the numerical features while we want to one-hot
# encode the categorical features. In this regard, we make use of the
# :class:`~sklearn.compose.ColumnTransformer`.

numerical_columns = [name for name in X_train.columns]
numerical_pipeline = make_pipeline(FunctionTransformer(func=convert_float64, validate=False), StandardScaler())

#categorical_columns = [name for name in X_train.columns if "_cat" in name]
#categorical_pipeline = make_pipeline(SimpleImputer(missing_values=-1, strategy="most_frequent"),OneHotEncoder(categories="auto"),)

preprocessor = ColumnTransformer(
    [
        ("numerical_preprocessing", numerical_pipeline, numerical_columns),
        #("categorical_preprocessing",categorical_pipeline,categorical_columns,),
    ],
    remainder="drop",
)

# Create an environment variable to avoid using the GPU. This can be changed.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.python import keras
from keras.layers import Activation, BatchNormalization, Dense, Dropout

###############################################################################
# Create a neural-network
###############################################################################
#from tensorflow.keras.models import Sequential
from keras.models import Sequential
import math


def make_model(n_features):
    feature_num = X_train.shape[1]  # 输入数据特征数
    #start_node = 10 * 2 ** (math.ceil(math.log(feature_num / 2, 2)))  # 第一层节点数量
    start_node = 200
    print(start_node)

    model = Sequential()
    model.add(Dense(start_node, input_shape=(n_features,), kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(Dropout(0.4))
    model.add(Dense(start_node/2, kernel_initializer="glorot_normal", use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(Dropout(0.3))
    model.add(Dense(start_node/4, kernel_initializer="glorot_normal", use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(Dropout(0.2))
    model.add(Dense(start_node/8, kernel_initializer="glorot_normal", use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(Dropout(0.1))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


###############################################################################
# We create a decorator to report the computation time

import time
from functools import wraps


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start_time = time.time()
        result = f(*args, **kwds)
        elapsed_time = time.time() - start_time
        print(f"Elapsed computation time: {elapsed_time:.3f} secs")
        return (elapsed_time, result)

    return wrapper


###############################################################################
# The first model will be trained using the ``fit`` method and with imbalanced
# mini-batches.
import tensorflow
from sklearn.metrics import roc_auc_score
from sklearn.utils import parse_version

tf_version = parse_version(tensorflow.__version__)


# 不均衡
@timeit
def fit_predict_imbalanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=50)
    if tf_version < parse_version("2.6"):
        # predict_proba was removed in tensorflow 2.6
        predict_method = "predict_proba"
    else:
        predict_method = "predict"
    y_pred = getattr(model, predict_method)(X_test, batch_size=50)
    #return roc_auc_score(y_test, y_pred)
    return (y_test, y_pred)


###############################################################################
# In the contrary, we will use imbalanced-learn to create a generator of
# mini-batches which will yield balanced mini-batches.

from imblearn.keras import BalancedBatchGenerator

@timeit
def fit_predict_balanced_model(X_train, y_train, X_test, y_test):
    model = make_model(X_train.shape[1])
    training_generator = BalancedBatchGenerator(X_train, y_train, batch_size=50, random_state=42)
    model.fit(training_generator, epochs=1000, verbose=1)
    y_pred = model.predict(X_test, batch_size=50)
    #return roc_auc_score(y_test, y_pred)
    return (y_test, y_pred)

###############################################################################
# Classification loop
###############################################################################

###############################################################################
# We will perform a 10-fold cross-validation and train the neural-network with
# the two different strategies previously presented.

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score

skf = StratifiedKFold(n_splits=10)

cv_results_imbalanced = []
cv_time_imbalanced = []
cv_results_balanced = []
cv_time_balanced = []
y_test_imbalanced = []
y_pred_imbalanced = []
y_test_balanced = []
y_pred_balanced = []

y_pred_pro_balanced = []
y_pred_pro_imbalanced = []

for train_idx, valid_idx in skf.split(X_train, y_train):
    X_local_train = preprocessor.fit_transform(X_train.iloc[train_idx])
    y_local_train = y_train.iloc[train_idx].values.ravel()
    X_local_test = preprocessor.transform(X_train.iloc[valid_idx])
    y_local_test = y_train.iloc[valid_idx].values.ravel()

    # elapsed_time, (y_test, y_pred) = fit_predict_imbalanced_model(X_local_train, y_local_train, X_local_test, y_local_test)
    # roc_auc = roc_auc_score(y_test, y_pred)
    # cv_time_imbalanced.append(elapsed_time)
    # cv_results_imbalanced.append(roc_auc)
    #
    # for i in y_test:
    #     y_test_imbalanced.append(i)
    # for i in y_pred:
    #     y_pred_pro_imbalanced.append(i)
    #     if i >0.5:
    #         y_pred_imbalanced.append(1)
    #     else:
    #         y_pred_imbalanced.append(0)

    elapsed_time, (y_test, y_pred) = fit_predict_balanced_model(X_local_train, y_local_train, X_local_test, y_local_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    cv_time_balanced.append(elapsed_time)
    cv_results_balanced.append(roc_auc)


    for i in y_test:
        y_test_balanced.append(i)
    for i in y_pred:
        y_pred_pro_balanced.append(i)
        if i >0.5:
            y_pred_balanced.append(1)
        else:
            y_pred_balanced.append(0)

# print(len(y_test_imbalanced))
# print(y_pred_imbalanced)
#
# print("EasyEnsemble classifier performance:")
# print(
#     f"Balanced accuracy: {balanced_accuracy_score(y_test_imbalanced, y_pred_imbalanced):.2f} - "
#     f"Geometric mean {geometric_mean_score(y_test_imbalanced, y_pred_imbalanced):.2f}"
# )
#
# print(confusion_matrix(y_test_imbalanced, y_pred_imbalanced))
#
# target_names = ['class 0', 'class 1']
# print(classification_report_imbalanced(y_test_imbalanced, y_pred_imbalanced, target_names=target_names))



print(len(y_test_balanced))
print(y_pred_balanced)

print("EasyEnsemble classifier performance:")
print(
    f"Balanced accuracy: {balanced_accuracy_score(y_test_balanced, y_pred_balanced):.2f} - "
    f"Geometric mean {geometric_mean_score(y_test_balanced, y_pred_balanced):.2f}"
)

print(confusion_matrix(y_test_balanced, y_pred_balanced))

target_names = ['class 0', 'class 1']
print(classification_report_imbalanced(y_test_balanced, y_pred_balanced, target_names=target_names))


# top k

A = y_pred_pro_balanced
B = y_test_balanced
# A = y_pred_pro_balanced
# B = y_test_balanced

Z = zip(A, B)
Z = sorted(Z, reverse=True)

A_new, B_new = zip(*Z)

for i in range(len(A_new)):
    if i in [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
        print(str(i) + ":" + str(sum(B_new[0:i])))

''''
target_names = ['class 0', 'class 1']
print(classification_report_imbalanced(y_test_imbalanced, y_pred_imbalanced, target_names=target_names))

ConfusionMatrixDisplay(confusion_matrix(y_test_imbalanced, y_pred_imbalanced)).plot()
plt.show()


target_names = ['class 0', 'class 1']
print(classification_report_imbalanced(y_test_balanced, y_pred_balanced, target_names=target_names))

ConfusionMatrixDisplay(confusion_matrix(y_test_balanced, y_pred_balanced)).plot()
plt.show()
'''''
###############################################################################
# Plot of the results and computation time
###############################################################################
'''
df_results = pd.DataFrame(
    {
        "Balanced model": cv_results_balanced,
        "Imbalanced model": cv_results_imbalanced,
    }
)
df_results = df_results.unstack().reset_index()

df_time = pd.DataFrame(
    {"Balanced model": cv_time_balanced, "Imbalanced model": cv_time_imbalanced}
)
df_time = df_time.unstack().reset_index()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.boxplot(y="level_0", x=0, data=df_time)
sns.despine(top=True, right=True, left=True)
plt.xlabel("time [s]")
plt.ylabel("")
plt.title("Computation time difference using a random under-sampling")

plt.figure()
sns.boxplot(y="level_0", x=0, data=df_results, whis=10.0)
sns.despine(top=True, right=True, left=True)
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
plt.xlabel("ROC-AUC")
plt.ylabel("")
plt.title("Difference in terms of ROC-AUC using a random under-sampling")
'''