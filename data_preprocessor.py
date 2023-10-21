from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def preprocess(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.4, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
