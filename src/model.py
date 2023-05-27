import lib.experimentize as E
from autosklearn.classification import AutoSklearnClassifier

class WithModel:
    def auto_classifier(self, *, time_limit=E.param, seed=E.param):
        return AutoSklearnClassifier(
            time_left_for_this_task=time_limit,
            include = {
                'feature_preprocessor': ["no_preprocessing"]
            },
            seed=seed
        )