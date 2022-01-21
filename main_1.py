from Model.LinearRegression import LR_model
from Model.DecisonTreeRegression import DTR_model
from Data.DataClean import GameData
from Data.train import Train_data
from Data.test import Test_data
from Predict_Analysis import Evaluate_Prediction
from FeatureSelection import DataSelection

'''
1.引用 class GameData, 幫忙做取得資料
2. 將得到的結果傳給 class Train_data, Test_data, 取得清理完的訓練集與測試集
3. 將結果傳給 class DataSelection, 得到特徵篩選後所選的特徵
4. 將結果傳給 class LR_model, DTR_model, 得到建模的模型與建模的R^2
5. 最終將結果傳給 class Evaluate_Prediction，得到兩個模型的評估與預測結果

'''
if __name__ == '__main__': 
    #讓使用者可以輸入隊伍的名稱
    input_team_name = input('Please type one team by its abbreviation:')
    #讓使用者可以輸入需要探討的年份
    try:
        input_game_year = input('Please type year by 1996 to 2021:')
        if int(input_game_year)>2021:
            raise 
        elif int(input_game_year)<1996:
            raise            
    except:
        print('Sorry, please type again.')
    
    #讓使用者輸入新增加特徵的場次
    try:
        input_past_games = int(input('Please type a nubmer less 10 to decide the past game of data rolling:'))
        if input_past_games>10:
            raise      
    except:
        print('Sorry, please type again.')

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
    training_data = Train_data(input_past_games, games)
    clean_train_data = training_data.rolling_data()
    print('Train data:')
    print(clean_train_data)

    #取得清理完的測試資料
    testing_data = Test_data(input_past_games, games)
    clean_test_data = testing_data.rolling_data()
    print("Test data:")
    print(clean_test_data)

    #要給FeatureSelection結果，才有辦法繼續跑動
    fs_data = DataSelection(clean_train_data, input_past_games)
    #透過DataSelection的物件，取得篩選資料的方法
    select_fectures = fs_data.RFECV_selection()
    print('Features:', select_fectures)

    #給LR_model結果，才有辦法繼續跑動
    lr_model_data = LR_model(clean_train_data, clean_test_data, select_fectures)
    #得知LR_model的R^2
    lin_mmodel = lr_model_data.model()
    
    #給DTR_model結果，才有辦法繼續跑動
    dtr_model_data = DTR_model(clean_train_data, clean_test_data, select_fectures)
    dtr_model = dtr_model_data.model()

    #給Predict_Analysis結果，才有辦法繼續跑動
    model_data_ = Evaluate_Prediction(clean_train_data, clean_test_data, select_fectures)
    cross_valid = model_data_.Cross_Valid_TSS()
   

    print('Linear Regression model:')
    prediction_lr = model_data_.Predict_lr()
    
    print('Decision Tree Regression model:')
    prediction_dtr = model_data_.Predict_dtr()
    
   # print(prediction_dtr)

   

