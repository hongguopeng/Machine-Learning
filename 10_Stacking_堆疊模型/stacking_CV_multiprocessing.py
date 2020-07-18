import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score , precision_score , roc_curve , precision_recall_curve , auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.base import clone
from sklearn.model_selection import KFold
import time
import warnings
warnings.filterwarnings("ignore")

seed = 10
np.random.seed(seed)
df = pd.read_csv('data.csv')
y = (df['cand_pty_affiliation'] == 'REP').astype(np.float32)
X = df.drop(['cand_pty_affiliation'] , axis = 1)
X = pd.get_dummies(X , sparse = True)

global x_train , y_train
x_train , x_test , y_train , y_test =\
train_test_split(X , y , test_size = 0.1 , random_state = seed)
x_train ,  y_train = np.array(x_train) , np.array(y_train)
x_test , y_test = np.array(x_test) , np.array(y_test)


def job(q , core , learners , train_idx , test_idx):
    global x_train , y_train
    x_train_fold , y_train_fold = x_train[train_idx , :] , y_train[train_idx]
    x_test_fold , y_test_fold = x_train[test_idx] , y_train[test_idx]

    fold_learners = {learner_name : clone(learner) for learner_name , learner in learners.items()}

    # 利用x_train_fold與y_train_fold再訓練每個弱分類器
    # 接著收集各個分類器針對x_test_fold的預測結果，並存進y_prob_first
    y_prob_first = []
    for j , (learner_name , learner) in enumerate(zip(fold_learners.keys() , fold_learners.values())):
        print('Stacking first , core : {} , inner_loop : {} , base learner : {}'.format(core , j , learner_name))
        learner.fit(x_train_fold , y_train_fold)
        y_prob = learner.predict_proba(x_test_fold)[: , 1]
        y_prob_first.append(y_prob)
    y_prob_first = np.array(y_prob_first).T

    q.put([y_prob_first , y_test_fold])


if __name__ == '__main__':

    knn = KNeighborsClassifier(n_neighbors = 2)
    nb = GaussianNB()
    rf = RandomForestClassifier(n_estimators = 3 , max_features = 3 , random_state = seed)
    gb = GradientBoostingClassifier(n_estimators = 100 , random_state = seed)
    lr = LogisticRegression(C = 50 , random_state = seed)
    learners = {'knn' : knn ,
                'naive bayes' : nb ,
                'random forest': rf ,
                'gbm': gb ,
                'logistic': lr}

    start = time.time()
    # Step1 ⇨ 針對x_train與y_train，先訓練一次弱分類器
    for i , (learner_name , learner) in enumerate(zip(learners.keys() , learners.values())):
        print('index : {} , base learner : {}'.format(i , learner_name))
        learner.fit(x_train , y_train)

    # Step2
    processes = []
    manager = mp.Manager()
    q = manager.Queue()
    for core , (train_idx , test_idx) in enumerate(KFold(4).split(x_train)):
        p = mp.Process(target = job , args = [q , core , learners , train_idx , test_idx])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    y_prob_cv , y_test_cv = [] , []
    for _ in range(0 , len(processes)):
        print('q.qsize() : ' , q.qsize())
        outcome = q.get()
        y_prob_cv.append(outcome[0])
        y_test_cv.append(outcome[1])

    y_prob_cv = np.vstack(y_prob_cv)
    y_test_cv = np.hstack(y_test_cv)

    # Step3 ⇨ 利用y_prob_cv與y_test_cv開始訓練second_learner
    second_learner = lgb.LGBMClassifier(num_leaves = 200 ,
                                        learning_rate = 0.005 ,
                                        n_estimators = 1000 ,
                                        max_depth = 4 ,
                                        random_state = seed ,
                                        n_jobs = -1)
    second_learner.fit(y_prob_cv , y_test_cv)

    # Step4 ⇨ 將x_test丟入弱分類器中產生初步預測結果
    y_prob_second = []
    for i , (learner_name , learner) in enumerate(zip(learners.keys() , learners.values())):
        print('Stacking second , index : {} , base learner : {}'.format(i , learner_name))
        y_prob = learner.predict_proba(x_test)[: , 1]
        y_prob_second.append(y_prob)
    y_prob_second = np.array(y_prob_second).T

    # Step5 ⇨ 再將弱分類器產生初步預測結果丟進second_learner中得到最終預測結果
    y_prob_final = second_learner.predict_proba(y_prob_second)[: , 1]

    precision_stack , recall_stack , _ = precision_recall_curve(y_test , y_prob_final)
    pr_auc_stack = auc(recall_stack , precision_stack)
    print('\nStacking_PR-AUC : {:.4f}'.format(pr_auc_stack))

    end = time.time()
    print('Operation Time : {}'.format(end - start))
