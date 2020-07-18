import pandas as pd
import numpy as np
import multiprocessing as mp

global sub_set , total_song , user_song
sub_set = pd.read_csv(filepath_or_buffer = './sub_set.csv' , encoding = 'ISO-8859-1')
user = 'a974fc428825ed071281302d6976f59bfa95fe7e' # 針對某一個用戶進行推薦
total_song = list(set(sub_set['title'])) # 取得sub_set中所有的歌曲
user_song = list(set(sub_set.loc[sub_set['user'] == user]['title'])) # 取得欲推薦用戶聽過的所有歌曲(此用戶共聽過66首歌曲)
total_song = sorted(total_song) # 排序以固定順序
user_song = sorted(user_song) # 排序以固定順序
# total_song = list(sub_set['title'].unique()) # 取得sub_set中所有的歌曲
# user_song = list(sub_set.loc[sub_set['user'] == user]['title'].unique()) # 取得欲推薦用戶聽過的所有歌曲(此用戶共聽過66首歌曲)

def job(q , core , index):
    global sub_set , total_song , user_song
    target_song = user_song[index[0] : index[1]]
    jaccard = np.zeros((len(target_song) , len(total_song)))

    for i , song_i in enumerate(target_song):
        for j , song_j in enumerate(total_song):
    
            print('進度 => core : {} , 欲推薦用戶聽過的歌曲 : {} , 所有用戶聽過的歌曲 : {}'.format(core , i , j))
    
            # 把聽過i這首歌的人取出來
            user_set_i = set(sub_set.loc[sub_set['title'] == song_i]['user'])
    
            # 把聽過j這首歌的人取出來
            user_set_j = set(sub_set.loc[sub_set['title'] == song_j]['user'])
    
            # user_set_i、user_set_j 取交集
            intersection = list(user_set_j.intersection(user_set_i))

            if len(intersection) != 0:
                # user_set_i、user_set_j 取聯集
                union = list(user_set_i.union(user_set_j))
        
                # 交集 / 聯集
                jaccard[i , j] = len(intersection) / len(union)
    
            # 假如沒有交集，也就不需要計算聯集了，可以加快計算速度
            elif len(intersection) == 0:
                jaccard[i , j] = 0

    q.put([core , jaccard])


if __name__ == '__main__':
    # Jaccard matrix總共有66個row
    # 將Jaccard matrix的row分為8組(因為有8個core)處理，每1組core要處理8個row，而最後2組core要處理10個row

    # 決定好每個core要負責的row
    # 例如[8 , 16] => row : 8 ~ 15
    index_list = [[0 , 8] , [8 , 16] , [16 , 24] , [24 , 32] ,
                  [32 , 40] , [40 , 48] , [48 , 56] , [56 , 66]]
    processes = []
    manager = mp.Manager()
    q = manager.Queue()
    for core , index in enumerate(index_list):
        p = mp.Process(target = job , args = [q , core , index])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    
    print('q.full()  : ' , q.full())
    outcome = []
    for _ in range(0 , len(processes)):
        outcome.append(q.get())
        print('q.qsize() : ' , q.qsize())

    # 將outcome按照core的順序排列
    # 可以讓Jaccard matrix的row的順序與user_song的順序相同
    outcome = sorted(outcome , key = lambda x : x[0])

    Jaccard = outcome[0][1]
    for i , jaccard in enumerate(outcome):
        if i > 0:
            Jaccard = np.vstack([Jaccard , jaccard[1]])

    # 將Jaccard matrix存檔
    np.save('Jaccard_matrix' , Jaccard)