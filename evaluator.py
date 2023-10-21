from sklearn.metrics import accuracy_score, classification_report

class Evaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluate(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        report = classification_report(self.y_true, self.y_pred)
        return accuracy, report
