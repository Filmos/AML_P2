import lib.experimentize as E
import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS = ["artificial"]

artificial = (
    pd.read_csv('data/artificial_train.data', delim_whitespace=True, header=None),
    pd.read_csv('data/artificial_train.labels', header=None),
    pd.read_csv('data/artificial_valid.data', delim_whitespace=True, header=None)
)

class DatasetLoader:
    def get_data(self, *, dataset=E.param):
        return {
            "artificial": artificial
        }[dataset]
    
    def train_test_split(self, *, seed=E.param):
        train_x, train_y, _ = self.get_data()
        return train_test_split(train_x, train_y, test_size=0.2, random_state=seed)