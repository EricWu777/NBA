from datacleaning.data_by_cleaning import DataByCleaning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import os

class TestData(DataByCleaning):

    
    def rolling_data(self):
        #games_data=GameData.get_games(self)
        #由於季後賽為4~6月，故將常規賽設為train_data, playoffs設為test_data
        test_data =self.games_data[self.games_data.SEASON_ID.str[0]=='4']
        #還可以修改程式，去建構例外處理，以避免當年無季後賽，則無Test_data

        #Data rolling 
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
        ###variable可以再改
        variable = ['PTS','FG_PCT', 'FGM',"FGA", "FG3A", "FTM", "FTA", "REB", "STL","BLK","TOV","PF"]
        test_data = test_data.sort_values(by=['GAME_DATE'])
        for i in range(0,len(variable)):
            test_data[rolling_data_value[self.pastgame_num-1][i]] = np.round(test_data[variable[i]].rolling(window=self.pastgame_num, center = False).mean(),2)

            test_data[player_rolling_data_value[self.pastgame_num-1][i]] = np.round(test_data[variable[i]].rolling(window=self.pastgame_num, center = False).mean(),2)

            #特徵工程 分類變數轉連續變數 獨熱編碼 標籤編碼 有序編碼

        #將主客場欄位改成0,1
        test_data['Home'] = test_data['Home'].astype(int)

        #將Game_ID轉換成數字型式
        test_data['GAME_ID'] = test_data['GAME_ID'].astype(int)

        #將SEASON_ID轉換成數字型式
        test_data['SEASON_ID'] = test_data['SEASON_ID'].astype(int)

        #將"WL"做編碼，並做出一個新欄位Last_WL
        size_mapping = {'W': 1,'L': 0}
        test_data['Last_WL'] = test_data['WL'].shift().map(size_mapping) 

        #將'Player'做標籤編碼
        test_data['Player'] = LabelEncoder().fit_transform(test_data['Player'])

        #使用均值替換法來填補遺失值
        imr = SimpleImputer(missing_values=np.nan, strategy='median')
        test_data.iloc[:,31:56] = imr.fit_transform(test_data.iloc[:,31:56])

        return test_data 