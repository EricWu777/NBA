from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class Prediction:
    
    def __init__(self, train_data, test_data, features, model ):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features
        self.model = model

    #預測前十場分數
    def predict(self):
        y = self.train_data['PTS']
        X = self.train_data[self.features]  
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42 ) 
        #model  = LinearRegseression().fit(X_train, y_train)
        predict_points = self.model.predict(X)[:10]
        answer = self.test_data['PTS']
        print('Predict points', predict_points)
        print('True points:', answer)

    # def Predict_dtr(self):
    #     y = self.train_data['PTS']
    #     X = self.train_data[self.features]  
    #     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42 ) 
    #     #dtr_model = DecisionTreeRegressor(max_depth=4)
    #     #model  = dtr_model.fit(X_train, y_train)
    #     predict_points = self.model.predict(X)[:10]
    #     answer = self.test_data['PTS']
    #     print('Predict points', predict_points)
    #     print('True points:', answer)
        