import lib.experimentize as E
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif, chi2

FEATURE_SELECTION_METHODS = ["Random Forest", "ANOVA", "Mutual Info", "Chi-Squared"]

def support_to_features(support, data):
    return [data.columns[i] for i, x in enumerate(support) if x]

class FeatureSelector:
    def feature_selection_random_forest(self, train_x, train_y, *, max_features=E.param, seed=E.param):
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(train_x, train_y)
        selector = SelectFromModel(clf, prefit=True, max_features=max_features)
        return lambda x: selector.transform(x), support_to_features(selector.get_support(), train_x)

    def feature_selection_anova(self, train_x, train_y, *, max_features=E.param):
        selector = SelectKBest(f_classif, k=max_features)
        selector.fit(train_x, train_y)
        return lambda x: selector.transform(x), support_to_features(selector.get_support(), train_x)

    def feature_selection_mutual_info(self, train_x, train_y, *, max_features=E.param):
        selector = SelectKBest(mutual_info_classif, k=max_features)
        selector.fit(train_x, train_y)
        return lambda x: selector.transform(x), support_to_features(selector.get_support(), train_x)

    def feature_selection_chi_squared(self, train_x, train_y, *, max_features=E.param):
        selector = SelectKBest(chi2, k=max_features)
        selector.fit(train_x, train_y)
        return lambda x: selector.transform(x), support_to_features(selector.get_support(), train_x)