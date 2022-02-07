import json
import os
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

class DataSelection:
    def __init__(self, train_data, pastgame_num):
        self.train_data = train_data
        #self.test_data = test_data
        self.pastgame_num = pastgame_num
    

    #RFECV篩選
    def RFECV_selection(self):
        y= self.train_data['PTS']
        # 放入可能的特徵欄位(透過DataCleaning的方法引入)
        #先透過json檔案取得變數的名稱，並透過使用者輸入的場次做data rolling
        path = os.getcwd()
        with open(path + '/resource/rollingdata.json', encoding='utf-8') as f:
            rolling_data = json.load(f)
        with open(path + '/resource/rollingdata_player.json', encoding='utf-8')as f:
            rolling_data_player = json.load(f)
        
        value = rolling_data.values()  
        rolling_data_value = list(value)

        player_value = rolling_data_player.values()     
        player_rolling_data_value = list(player_value)

        cols = rolling_data_value[self.pastgame_num-1]
        cols_player = player_rolling_data_value[self.pastgame_num-1]
        # 將兩個list合併，並多放入兩個特徵
        cols.extend(cols_player)
        cols.extend(['Home','Last_WL','Player'])
        
        X = self.train_data[cols]
        
        rfecv = RFECV(LinearRegression(), step=3 , cv=5)
        rfecv.fit(X,y)
        features = [f for f, s in zip(X, rfecv.support_) if s]
        return features

