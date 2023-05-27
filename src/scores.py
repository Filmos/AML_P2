import lib.experimentize as E
from sklearn import metrics

class Scoring:
    def get_scores(self, test_y, pred_y, features, *, dataset=E.param):
        accuracy = metrics.balanced_accuracy_score(test_y, pred_y)
        feature_factor = {
            "artificial": 1/5,
            "spam": 1/100
        }[dataset]

        return {
            "Accuracy": accuracy,
            "Score": accuracy - 0.01*max(0, feature_factor * features - 1)
        }

