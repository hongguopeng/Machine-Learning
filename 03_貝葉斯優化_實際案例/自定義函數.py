import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score , recall_score , precision_score , precision_recall_curve , roc_curve , auc
from sklearn.model_selection import train_test_split

# 繪圖
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 建模
import lightgbm as lgb

data = pd.read_csv('data/caravan-insurance-challenge.csv')
train = data.loc[data['ORIGIN'] == 'train']
test = data.loc[data['ORIGIN'] == 'test']

# Extract the labels and format properly
train_labels = np.array(train['CARAVAN'])
test_labels = np.array(test['CARAVAN'])

# Drop the unneeded columns
train = train.drop(['ORIGIN' , 'CARAVAN'] , axis = 1)
test = test.drop(['ORIGIN' , 'CARAVAN'] , axis = 1)

# Convert to numpy array for splitting in cross validation
features = np.array(train) # features -> training data
test_features = np.array(test)
labels = train_labels[:]

# 使用LightGBM預設參數進行建模
model = lgb.LGBMClassifier()

# 創造交叉驗店驗證的dataset
train_set = lgb.Dataset(features , label = labels) 

# Hyperparameter grid
param_grid = {'class_weight' : [None , 'balanced'],
              'boosting_type' : ['gbdt' , 'goss'],
              'num_leaves' : list(range(30 , 150)),
              'learning_rate' : list(np.logspace(np.log(0.005) , np.log(0.2) , base = np.exp(1) , num = 1000)),
              'subsample_for_bin' : list(range(20000 , 300000 , 20000)),
              'min_child_samples' : list(range(20 , 500 , 5)),
              'reg_alpha' : list(np.linspace(0 , 1)),
              'reg_lambda' : list(np.linspace(0 , 1)),
              'colsample_bytree' : list(np.linspace(0.6 , 1 , 10))}

# 把subsample_dist獨立出來的原因是，因為boosting_type若是'goss'的話，那subsample一定要是1
# 另外2種則不受限
subsample_dist = list(np.linspace(0.5 , 1 , 100))

random.seed(50)
params = {}
for key in param_grid.keys():
    params[key] = random.sample(param_grid[key] , 1)[0]

if params['boosting_type'] != 'goss':
    params['subsample'] = random.sample(subsample_dist , 1)[0]  
else :
    params['subsample'] = 1.0

#  ⇑
#  ⇑  以上都不太重要
#--------------------------------------------------------------------------------------------------------------------#
#  ⇓  重要的是以下怎麼自定義函數
#  ⇓

X_train , X_val , y_train , y_val = train_test_split(train , train_labels , test_size = 0.3 , random_state = 0)
train_data = lgb.Dataset(X_train, label = y_train)
val_data = lgb.Dataset(X_val , label = y_val)

# 自定義評價函數
# 注意需要返回三個參數
# 最後一個參數為True，則metric越大越好；最後一個參數為False，則metric越小越好
# prob = np.argmax(prob.reshape(len(label) , -1) , axis = 1) # for multiclass
def evaluate_metrics(prob , data):
    global metrics , belta
    label = data.get_label()
    
    # 在此是針對binary classification
    # 若是要針對multi-classfication，在計算recall與precision都需要指定average的方法:micro、macro、weighted
    prob = (prob > 0.5).astype(np.int32)

    if metrics == 'recall':
        recall = recall_score(label , prob)
        return metrics , recall , True

    elif metrics == 'precision':
        precision = precision_score(label , prob)
        return metrics , precision , True

    elif metrics == 'f':
        recall = recall_score(label , prob)
        precision = precision_score(label , prob)
        f_score = (1 + belta ** 2) * precision * recall / ((belta ** 2) * precision + recall)
        return metrics + '{}'.format(belta) , f_score , True

global metrics , belta
metrics , belta = 'f' , 1
evals_result = {}
clf = lgb.train(params ,
                train_data ,
                num_boost_round = 10000 ,
                valid_sets = val_data ,
                valid_names = 'validation',
                early_stopping_rounds = 200 ,
                feval = evaluate_metrics ,
                evals_result = evals_result ,
                verbose_eval = False ,
                seed = 50)
evals_result['validation'][metrics + '{}'.format(belta)]


# 自定義評價函數
# 注意需要返回三個參數
# 最後一個參數為True，則metric越大越好；最後一個參數為False，則metric越小越好
def compute_auc(prob , data):
    global curve
    label = data.get_label()
    if curve == 'ROC':
        fpr , tpr , _ = roc_curve(label , prob)
        auc_score = auc(fpr , tpr)
    elif curve == 'PR':
        precision , recall , _ = precision_recall_curve(label , prob)
        auc_score = auc(recall , precision)
    return 'AUC' , auc_score , True

global curve
curve = 'PR'
clf_cv = lgb.cv(params ,
                train_data ,
                num_boost_round = 10000 ,
                nfold = 10 ,
                feval = compute_auc ,
                early_stopping_rounds = 100 , # early_stopping_rounds = 100：如果再連續迭代100次還是沒進步，那就停止迭代
                verbose_eval = False ,
                seed = 50)
clf_cv_best = np.max(clf_cv['AUC-mean'])








































