from sklearn import datasets

class Dataset:
    def __init__(self):
        self.data, self.target = datasets.load_iris(return_X_y=True)
