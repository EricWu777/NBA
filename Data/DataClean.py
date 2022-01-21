from nba_api.stats.static import teams 
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json


class GameData:
    
    def __init__(self, team_name, game_year, pastgame_num):
        self.team_name = team_name
        self.game_year = game_year
        self.pastgame_num = pastgame_num

    #定義取得team_id的方法
    def get_team_id(self):
        team_id  = teams.find_team_by_abbreviation(str(self.team_name))['id']
        return team_id

    # def output(self):
    #    print(self.team_name)
    #    print(self.game_year)
    #    print(self.pastgame_num)

    #定義取得滿足使用者輸入的球賽
    def get_games(self):

        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=GameData.get_team_id(self))
        games = gamefinder.get_data_frames()[0]
        games_years = games[games.SEASON_ID.str[-4:] == self.game_year] 
        #去掉遺失值
        games_years = games_years[games_years.loc[:,'PTS'] != 0]
        games_years = games_years.dropna()
        # 定義一個新的欄位 year (是哪一年的比賽嗎?)
        games_years['year'] = games_years.SEASON_ID.str[1:5].astype(int)
        # 定義一個新的欄位 Home (是主場嗎?)
        games_years['Home'] = games_years.MATCHUP.str.len()>10
        # 定義一個新的欄位 Player (對手是?)
        games_years['Player'] = games_years.MATCHUP.str.rsplit(' ').str[-1]
        
        return games_years

#     def rolling_data(self):
#         games_data=GameData.get_games(self)
#         #由於季後賽為4~6月，故將常規賽設為train_data, playoffs設為test_data
#         train_data = games_data[(games_data.SEASON_ID.str[0]!='4') & (games_data.SEASON_ID.str[0]!='5')
#                                & (games_data.SEASON_ID.str[0]!='6')]
#         test_data = games_data[games_data.SEASON_ID.str[0]=='4']
#         #還可以修改程式，去建構例外處理，以避免當年無季後賽，則無Test_data

#         #Data rolling 
#         #先透過json檔案取得變數的名稱，並透過使用者輸入的場次做data rolling
#         with open('rollingdata.json') as f:
#             rolling_data = json.load(f)
#         with open('rollingdata_player.json')as f:
#             rolling_data_player = json.load(f)
     
#         value = rolling_data.values()
#         rolling_data_value = list(value)
        
#         player_value = rolling_data_player.values()
#         player_rolling_data_value = list(player_value)  
#         ###variable可以再改
#         variable = ['PTS','FG_PCT', 'FGM',"FGA", "FG3A", "FTM", "FTA", "REB", "STL","BLK","TOV","PF"]
#         train_data = train_data.sort_values(by=['GAME_DATE'])
#         test_data = test_data.sort_values(by=['GAME_DATE'])
#         for i in range(0,len(variable)):
#             train_data[rolling_data_value[self.pastgame_num-1][i]] = np.round(train_data[variable[i]].rolling(window=self.pastgame_num, center = False).mean(),2)
#             test_data[rolling_data_value[self.pastgame_num-1][i]] = np.round(test_data[variable[i]].rolling(window=self.pastgame_num, center = False).mean(),2)

#             train_data[player_rolling_data_value[self.pastgame_num-1][i]] = np.round(train_data[variable[i]].rolling(window=self.pastgame_num, center = False).mean(),2)
#             test_data[player_rolling_data_value[self.pastgame_num-1][i]] = np.round(test_data[variable[i]].rolling(window=self.pastgame_num, center = False).mean(),2)
# #for var in variable
# # for var in variable:
# #     Train_data[Rolling_data_value[PastGames_num-1][var]] = np.round(Train_data[variable[var]].rolling(window=PastGames_num, center = False).mean(),2)
# #     Test_data[Rolling_data_value[PastGames_num-1][var]] = np.round(Test_data[variable[var]].rolling(window=PastGames_num, center = False).mean(),2)

# #     Train_data[Player_rolling_data_value[PastGames_num-1][var]] = np.round(Train_data[variable[var]].rolling(window=PastGames_num, center = False).mean(),2)
# #     Test_data[Player_rolling_data_value[PastGames_num-1][var]] = np.round(Test_data[variable[var]].rolling(window=PastGames_num, center = False).mean(),2)
# # #檢查遺失值
# # print(Train_data.isnull().sum(),Train_data.shape)
# # print(Test_data.isnull().sum(),Test_data.shape)

#             #特徵工程 分類變數轉連續變數 獨熱編碼 標籤編碼 有序編碼

#             #將主客場欄位改成0,1
#         train_data['Home'] = train_data['Home'].astype(int)
#         test_data['Home'] = test_data['Home'].astype(int)

#         #將Game_ID轉換成數字型式
#         train_data['GAME_ID'] = train_data['GAME_ID'].astype(int)
#         test_data['GAME_ID'] = test_data['GAME_ID'].astype(int)

#         #將SEASON_ID轉換成數字型式
#         train_data['SEASON_ID'] = train_data['SEASON_ID'].astype(int)
#         test_data['SEASON_ID'] = test_data['SEASON_ID'].astype(int)

#         #將"WL"做編碼，並做出一個新欄位Last_WL
#         size_mapping = {'W': 1,'L': 0}

#         train_data['Last_WL'] = train_data['WL'].shift().map(size_mapping) 
#         test_data['Last_WL'] = test_data['WL'].shift().map(size_mapping) 

#         #將'Player'做標籤編碼
#         train_data['Player'] = LabelEncoder().fit_transform(train_data['Player'])
#         test_data['Player'] = LabelEncoder().fit_transform(test_data['Player'])

#         #使用均值替換法來填補遺失值
#         imr = SimpleImputer(missing_values=np.nan, strategy='median')
#         train_data.iloc[:,31:56] = imr.fit_transform(train_data.iloc[:,31:56])
#         test_data.iloc[:,31:56] = imr.fit_transform(test_data.iloc[:,31:56])

#         return train_data 