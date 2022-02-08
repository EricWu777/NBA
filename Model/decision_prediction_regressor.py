from model.machine_learning import MachineLearning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class DTRModel(MachineLearning):
    #__init__ 使用MachineLearning父類別的變數
    #overriding覆寫掉父類別的方法
    def model(self):
        y = self.train_data['PTS']
        X = self.train_data[self.features]  
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42 ) 

        DTR = DecisionTreeRegressor(max_depth=4)
        DTR.fit(X_train, y_train)
        self.show('Decision', DTR ,X_train,y_train, X_valid, y_valid)
       
        return DTR

        

    