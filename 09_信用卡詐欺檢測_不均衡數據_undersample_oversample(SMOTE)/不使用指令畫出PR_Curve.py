import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 讀取數據
data = pd.read_csv('creditcard.csv')

# 數據預處理
data['norm_Amount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1 , 1))
data = data.drop(['Time' , 'Amount'] , axis = 1)

# 將數據分為訓練集與測試集
y = np.array(data['Class']).astype(np.float32) # label
X = np.array(data.drop('Class' , axis = 1)).astype(np.float32) # feature
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 0)

# 以SMOTE平衡數據標籤
oversampler = SMOTE(random_state = 0)
X_train_oversample , y_train_oversample = oversampler.fit_sample(X_train , y_train)

# 建模
logstic_oversample = LogisticRegression(C = 10 , penalty = 'l1' , n_jobs = -1 , random_state = 0)
logstic_oversample.fit(X_train_oversample , y_train_oversample)

# 得到輸出正樣本的機率
y_prob_oversample = logstic_oversample.predict_proba(X_test)[:,1]

#  ⇑
#  ⇑  以上都不太重要
#--------------------------------------------------------------------------------------------------------------------#
#  ⇓  重要的是以下怎麼繪製PR Curve與計算PR Curve的AUC Score
#  ⇓  而繪製ROC Curve與計算ROC Curve的AUC Score的方法跟以下的流程差不多

threshold = np.sort(y_prob_oversample)[::-1] # 以輸出正樣本的機率當作閥值並由大到小作排序
recall_oversample , precision_oversample = [] , [] # 紀錄每個不同的閥值下的Recall與Precision
auc = 0
positive_sample_num = (y_test == 1).sum() # 得到真正正樣本數據的數據量
for i in tqdm(range(0 , len(threshold))):

    # 輸出正樣本的機率 => 大於閥值的記為1 小於閥值的記為0
    y_pred_oversample = (y_prob_oversample > threshold[i]).astype(np.int8)

    tp = ((y_test == 1) & (y_pred_oversample == 1)).sum() # 計算True Positive的數據量
    fp = ((y_test == 0) & (y_pred_oversample == 1)).sum() # 計算False Positive的數據量

    recall = tp / positive_sample_num
    precision = tp / (tp + fp)

    recall_oversample.append(recall)
    if np.isnan(precision) == True:
        precision_oversample.append(1)
    elif np.isnan(precision) == False:
        precision_oversample.append(precision)

    # precision在閥值最大的時候不會預測出正樣本，tp + fp一定是0
    # 所以此時precision為nan，需要將之過濾
    if np.isnan(precision) == False:
        auc += (recall_oversample[i] - recall_oversample[i - 1]) * precision

# 繪製PR Curve
fig , ax = plt.subplots(1 , 1 , figsize = (20 , 10))
ax.plot(recall_oversample , precision_oversample , lw = 2 , color = 'blue' , label = 'PR_Curve_oversample , AUC = {:.2f}'.format(auc))
ax.set_xlabel('Recall' , fontsize = 30)
ax.set_ylabel('Precision' , fontsize = 30)
ax.legend(loc = 'best' , fontsize = 20)



