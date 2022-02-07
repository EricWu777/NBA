import numpy as np

class MachineLearning:

    def __init__(self, train_data, test_data, features ):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features

        self.y = self.train_data['PTS']
        self.x = self.train_data[self.features]  
    
    def show(self, name, module, X_train,y_train, X_valid, y_valid):
        print(name, ' Tree Regression:')
        print('Traindata_R2 :' , module.score(X_train,y_train))
        print('Validdata_R2 :', module.score(X_valid, y_valid))

    def model(self):
        pass