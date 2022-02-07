from model.machine_learning import MachineLearning
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class LRModel(MachineLearning):
#建立迴歸模型
#考慮到label為連續值、資料量較少，且資料的特徵與label之間有線性關係       
    def model(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.x, self.y, 
                                            test_size=0.2, random_state=42 ) 
        lin_mod =LinearRegression()     
        lin_mod.fit(X_train,y_train)
        #顯示截距 / 斜率 /  R2 score / 欄位名稱 
        print("LinearRegression:")
        print('Intercept:' , lin_mod.intercept_)
        print('Coef:' , lin_mod.coef_)

        self.show('LinearRegression',lin_mod, X_train,y_train, X_valid, y_valid)
       

        
