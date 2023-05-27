import lib.experimentize as E
from src.feature_selector import FeatureSelector, FEATURE_SELECTION_METHODS
from src.dataset_loader import DatasetLoader, DATASETS
from src.model import WithModel
from src.scores import Scoring

class Params(E.ParamsBase):
    time_limit = 30
    dataset = DATASETS[0]
    method = FEATURE_SELECTION_METHODS[0]
    max_features = 1000
    seed = 1

class Core(FeatureSelector, DatasetLoader, WithModel, Scoring):
    pass