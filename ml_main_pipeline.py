from dataset import Dataset
from data_preprocessor import DataPreprocessor
from ml_model import MLModel
from evaluator import Evaluator

class MLMainPipeline:
    def run_pipeline(self):
        dataset = Dataset()
        data_preprocessor = DataPreprocessor(dataset.data, dataset.target)
        X_train, X_test, y_train, y_test = data_preprocessor.preprocess()

        ml_model = MLModel()
        ml_model.train(X_train, y_train)

        y_pred = ml_model.predict(X_test)

        evaluator = Evaluator(y_test, y_pred)
        accuracy, report = evaluator.evaluate()

        return accuracy, report
