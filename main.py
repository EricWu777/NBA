from datacleaning.test_data import TestData
from datacleaning.train_data import TrainData
from evaluation import Evaluation
from feature_selection import DataSelection
from gamedata import GameData
from model.decision_prediction_regressor import DTRModel
from model.linear_regression import LRModel
from predict import Prediction

'''
1.引用 class GameData, 幫忙做取得資料
2. 將得到的結果傳給 class Train_data, Test_data, 取得清理完的訓練集與測試集
3. 將結果傳給 class DataSelection, 得到特徵篩選後所選的特徵
4. 將結果傳給 class LR_model, DTR_model, 得到建模的模型與建模的R^2
5. 最終將結果傳給 class Evaluate_Prediction，得到兩個模型的評估與預測結果

'''
#讓使用者輸入要探討的年份
def get_input_game_year():
    try:
        input_game_year = input('Please type year by 1996 to 2021:')
        if int(input_game_year)>2021:
            raise 
        elif int(input_game_year)<1996:
            raise            
    except:
        print('Sorry, please type again.')
    return input_game_year

#讓使用者輸入新增加特徵的場次
def get_input_past_games():
    try:
        input_past_games = int(input('Please type a nubmer less 10 to decide the past game of data rolling:'))
        if input_past_games>10:
            raise      
    except:
        print('Sorry, please type again.')
    return input_past_games

def get_clean_train_data(input_past_games, games):
    training_data = TrainData(input_past_games, games)
    clean_train_data = training_data.rolling_data()
    print('Train data:')
    print(clean_train_data)
    return clean_train_data

def get_clean_test_data(input_past_games, games):
    testing_data = TestData(input_past_games, games)
    clean_test_data = testing_data.rolling_data()
    print("Test data:")
    print(clean_test_data)
    return clean_test_data


def get_select_fectures(input_past_games, clean_train_data):
    #要給FeatureSelection結果，才有辦法繼續跑動
    fs_data = DataSelection(clean_train_data, input_past_games)
    #透過DataSelection的物件，取得篩選資料的方法
    select_fectures = fs_data.RFECV_selection()
    print('Features:', select_fectures)
    return select_fectures

if __name__ == '__main__': 
    
    #讓使用者可以輸入隊伍的名稱
    input_team_name = input('Please type one team by its abbreviation:')
    
    input_game_year = get_input_game_year()
    
    input_past_games = get_input_past_games()

    #透過GameData物件，取得使用者輸入的比賽資料
    game_data = GameData(input_team_name, input_game_year, input_past_games)
    
    #取得team_id
    id = game_data.get_team_id()
    print('ID:', id)

    #取得game的資料
    games = game_data.get_games()
    print('Game data:')
    print(games)
   
    #取得清理完的訓練資料
    clean_train_data = get_clean_train_data(input_past_games, games)

    #取得清理完的測試資料
    clean_test_data = get_clean_test_data(input_past_games, games)

    #取得特徵篩選所得的特徵
    select_fectures = get_select_fectures(input_past_games, clean_train_data)

    #給LRModel結果，才有辦法繼續跑動
    lr_model_data = LRModel(clean_train_data, clean_test_data, select_fectures)
    #得知LRModel的R^2
    lin_mmodel = lr_model_data.model()
    
    #給DTRModel結果，才有辦法繼續跑動
    dtr_model_data = DTRModel(clean_train_data, clean_test_data, select_fectures)
    dtr_model = dtr_model_data.model()

    #給Evaluation結果，才有辦法繼續跑動

    #需要再把Evaluation的init_variable做修正
    eval_lr_data = Evaluation(clean_train_data, clean_test_data, select_fectures, lin_mmodel)
    cross_valid_by_lr = eval_lr_data.cross_valid_tss()

    eval_dtr_data = Evaluation(clean_train_data, clean_test_data, select_fectures, dtr_model)
    cross_valid_by_dtr = eval_dtr_data.cross_valid_tss()

    data_lr = Prediction(clean_train_data, clean_test_data, select_fectures, lin_mmodel)
    print('Linear Regression model:')
    prediction_lr = data_lr.predict()
    
    data_dtr = Prediction(clean_train_data, clean_test_data, select_fectures, dtr_model)
    print('Decision Tree Regression model:')
    prediction_dtr = data_dtr.predict()
    
   # print(prediction_dtr)

   
