from sklearn.svm import SVC

class MLModel:
    def __init__(self):
        self.model = SVC(kernel='linear', random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
