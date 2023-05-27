import lib.experimentize as E
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

DATASETS = ["artificial", "spam"]

artificial = (
    pd.read_csv('data/artificial_train.data', delim_whitespace=True, header=None),
    pd.read_csv('data/artificial_train.labels', header=None),
    pd.read_csv('data/artificial_valid.data', delim_whitespace=True, header=None)
)

spam = pd.read_csv("data/sms_train.csv")
def encode_spam(spam):
    vec = CountVectorizer(min_df=3, strip_accents="ascii")
    vec.fit(spam[0])
    transform = lambda x: pd.DataFrame(vec.transform(x).toarray(), columns=vec.get_feature_names())
    return (
        transform(spam[0]),
        spam[1],
        transform(spam[2])
    )
spam = encode_spam((
    spam["message"],
    spam["label"], 
    pd.read_csv("data/sms_test.csv")["message"]
))


class DatasetLoader:
    def get_data(self, *, dataset=E.param):
        return {
            "artificial": artificial,
            "spam": spam
        }[dataset]
    
    def train_test_split(self, *, seed=E.param):
        train_x, train_y, _ = self.get_data()
        return train_test_split(train_x, train_y, test_size=0.2, random_state=seed)