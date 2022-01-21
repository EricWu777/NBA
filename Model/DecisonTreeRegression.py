from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from Model.LinearRegression import LR_model


class DTR_model(LR_model):
    #__init__ 使用LR_model父類別的變數

    #overriding覆寫掉父類別的方法
    def model(self):
        y = self.train_data['PTS']
        X = self.train_data[self.features]  
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42 ) 

        DTR = DecisionTreeRegressor(max_depth=4)
        DTR.fit(X_train, y_train)
        
        print('Decision Tree Regression:')
        print('Traindata_R2 :' , DTR.score(X_train,y_train))
        print('Validdata_R2 :', DTR.score(X_valid, y_valid))
       

        

    