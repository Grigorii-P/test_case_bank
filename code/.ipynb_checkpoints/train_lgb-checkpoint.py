# add stratified
# сделать cross-val

import pandas as pd
import lightgbm as lgb
import json


path_data = '../tables/train_df_5.csv'
path_save_model = '../models/lgb_test.model'


def run_lgb(train_X, train_y, test_X, test_y, val_X=None, val_y=None):
    params = {
        "objective" : "multiclass",
        "num_classes": 80,
        "num_threads": 4,
        "max_depth": 25, # 6
        "learning_rate" : 0.03,
#         "num_leaves" : 30,
        "early_stopping_round": 50,
        "metric" : "multi_logloss",
#         "min_child_samples" : 100,
        "verbosity" : -1,
        "bagging_seed" : 2018,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "num_iterations": 150
    }
    
    train = lgb.Dataset(train_X, label=train_y)
    test = lgb.Dataset(test_X, label=test_y)
    
    verb_eval=5
    model = lgb.train(params, train, valid_sets=[test], # valid_sets=[train, val], 
                      valid_names = ['test'],
                      verbose_eval=verb_eval)
    return model


data = pd.read_csv(path_data)
data = data.sample(frac=1)
print('Data has been loaded...')
data.drop(['hash_inn'], 1, inplace=True)
data = data[data['target'] != -1]
train_cols = [x for x in list(data.columns) if x != 'target']

X = data[train_cols]
y = data['target']

p = 0.9
train_portion = int(X.shape[0] * p)
X_train, y_train = X.iloc[:train_portion], y.iloc[:train_portion]
X_test, y_test = X.iloc[train_portion:], y.iloc[train_portion:]

X_test.to_csv('X_test.csv')
y_test.to_csv('y_test.csv')

model = run_lgb(X_train, y_train, X_test, y_test)
model.save_model(path_save_model)
