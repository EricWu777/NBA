from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#時間序列交叉驗證
class Evaluation:
    def __init__(self, train_data, test_data, features ):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features

    def Cross_Valid_TSS_by_LR(self):
        y = self.train_data['PTS']
        X = self.train_data[self.features]  
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42 ) 

        tscv = TimeSeriesSplit(n_splits=5)

        R2 = cross_val_score(LinearRegression(), X_train, y_train, cv=tscv, scoring='r2')
        print('R2 by Cross Validation by TimeSeriesSplit', R2)
        print(f"\nR2: {R2.mean()} (+/- {R2.std()}")

        mse = cross_val_score(LinearRegression(), X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        print('MSE by Cross Validation by TimeSeriesSplit', mse)
        print(f"negative_MSE: {mse.mean()} (+/- {mse.std()}")
