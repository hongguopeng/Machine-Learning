import pandas as pd
import numpy as np
import multiprocessing as mp

sub_set = pd.read_csv(filepath_or_buffer = './sub_set.csv' , encoding = 'ISO-8859-1')
user = 'a974fc428825ed071281302d6976f59bfa95fe7e' # 針對某一個用戶進行推薦
total_song = list(set(sub_set['title'])) # 取得sub_set中所有的歌曲
user_song = list(set(sub_set.loc[sub_set['user'] == user]['title'])) # 取得欲推薦用戶聽過的所有歌曲(此用戶共聽過66首歌曲)
total_song = sorted(total_song) # 排序以固定順序
user_song = sorted(user_song) # 排序以固定順序

Jaccard_matrix = np.load('Jaccard_matrix.npy')

score = Jaccard_matrix.mean(axis = 0)
rank = score.argsort()[::-1] # score由大排到小的index全部取出來

for i in range(0 , 20):
    print(total_song[rank[i]])
