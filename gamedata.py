from nba_api.stats.static import teams 
from nba_api.stats.endpoints import leaguegamefinder

class GameData:
    
    def __init__(self, team_name, game_year, pastgame_num):
        self.team_name = team_name
        self.game_year = game_year
        self.pastgame_num = pastgame_num

    #定義取得team_id的方法
    def get_team_id(self):
        team_id  = teams.find_team_by_abbreviation(str(self.team_name))['id']
        return team_id

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

    def rolling_data(self):
            pass
