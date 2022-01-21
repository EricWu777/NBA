import numpy as np
from sklearn.linear_model import LinearRegression
#分拆數據集 for train valid
from sklearn.model_selection import train_test_split


class LR_model:
    def __init__(self, train_data, test_data, features ):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features
    
    
#建立迴歸模型
#考慮到label為連續值、資料量較少，且資料的特徵與label之間有線性關係
        
    def model(self):
        y = self.train_data['PTS']
        X = self.train_data[self.features]  
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42 ) 

    
        lin_mod =LinearRegression()     
        lin_mod.fit(X_train,y_train)
        #顯示截距 / 斜率 /  R2 score / 欄位名稱 
        print("LinearRegression:")
        print('Intercept:' , lin_mod.intercept_)
        print('Coef:' , lin_mod.coef_)
        print('Traindata_R2 :' , lin_mod.score(X_train,y_train))
        print('Validdata_R2 :', lin_mod.score(X_valid, y_valid))
        #print(LinearRegression().intercept_)
        #print(LinearRegression().coef_)
        #LinearRegression().score()
        

        
