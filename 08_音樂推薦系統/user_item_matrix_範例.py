import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

ex = pd.DataFrame({'user':['Kevin' , 'Tom' , 'Mary' , 'Kevin' , 'Jane' , 'Chris' , 'Joy' , 'Tom' , 'Bird'] , 
                   'item':['出口' , '馬戲團運動' , '馬戲團運動' , '麋途' , '天黑' , '黃昏市長' , '麋途' , '沒名字的人類' , '出口'] ,
                   'score':[10 , 9 , 7 , 6 , 8 , 6 , 4 , 3 , 5]})

user = ex['user'].drop_duplicates().reset_index()
user['user_index'] = user.index

item = ex['item'].drop_duplicates().reset_index()
item['item_index'] = item.index

ex_ = pd.merge(left = ex , right = user , on = 'user'  , how = 'left')
ex_ = pd.merge(left = ex_ , right = item , on = 'item' , how = 'left')
ex_ = ex_.sort_values(by = 'user_index').reset_index(drop = True)

score = np.array(ex_.score)
user_index = np.array(ex_.user_index)
item_index = np.array(ex_.item_index)
coo_matrix((score , (user_index , item_index))).toarray()