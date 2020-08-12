from ScamCall.filter_features import bool_feature, num_feature, stack, train_split, analysis_path, svm_cross_validation, score
from ScamCall.baseline089 import feats
from ScamCall.analysis_all_feature import main
from ScamCall.analysis_best import eda
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import xgboost as xgb


def merge_df(df_list):
    new_df = pd.merge(df_list[0], df_list[1], on='phone_no_m', how='outer')
    for i in range(2, len(df_list)):
        new_df = pd.merge(new_df, df_list[i], on='phone_no_m', how='outer')
    return new_df


def generate_all_feature():
    train1, test1 = main()
    train2, test2 = feats()
    train3, test3 = bool_feature('train'), bool_feature('test')
    train4, test4 = num_feature('train'), num_feature('test')

    train_df_all = merge_df([train1, train2, train3, train4])
    test_df_all = merge_df([test1, test2, test3, test4])

    train_df_all.to_csv(analysis_path + 'all_features_train.csv', index=False)
    test_df_all.to_csv(analysis_path + 'all_features_test.csv', index=False)

    return train_df_all, test_df_all


def svm_model(x_train, x_test, y_train, y_test):
    svm_m = svm_cross_validation(x_train, y_train)
    pre = svm_m.predict(x_test)
    s = score(pre, y_test)
    return s


def lr_model(x_train, x_test, y_train, y_test):
    lr = LogisticRegression()
    clf = lr.fit(x_train, y_train)
    pre = clf.predict(x_test)
    s = score(pre, y_test)
    return s


def gbdt_model(x_train, x_test, y_train, y_test):
    gbm0 = GradientBoostingClassifier(random_state=2020)
    gbm0.fit(x_train, y_train)
    pre = gbm0.predict(x_test)
    s = score(pre, y_test)
    return s


def xgb_model(x_train, x_test, y_train, y_test):
    xgb_val = xgb.DMatrix(x_test, label=y_test)
    xgb_train = xgb.DMatrix(x_train, label=y_train)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': True,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.01,  # 如同学习率
        'n_jobs': -1,  # cpu 线程数
    }

    plst = list(params.items())
    num_rounds = 5000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=200)
    return model


def cb_model(X_t, X_v, y_t, y_v):
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, loss_function='Logloss',
                               logging_level='Verbose', eval_metric='F1')
    model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=200, silent=True)
    return model.best_score_['validation']['F1']


def _train(columns, dis_col, col, train_df, model):
    train_c = list(set(columns).difference(set(dis_col + [col])))  # 去掉负影响特征
    train_c.append('label')
    train_c.append('phone_no_m')
    x_train, x_test, y_train, y_test = train_split(train_df[train_c])
    _score = model(x_train, x_test, y_train, y_test)
    return _score


def elimination(train_df):
    columns = train_df.columns.tolist()
    columns.remove('phone_no_m')
    columns.remove('label')

    cb_col = []  # 负影响或无影响特征
    svm_col = []
    lr_col = []
    gbdt_col = []

    x_train, x_test, y_train, y_test = train_split(train_df)
    cb_best = cb_model(x_train, x_test, y_train, y_test)
    svm_best = svm_model(x_train, x_test, y_train, y_test)
    lr_best = lr_model(x_train, x_test, y_train, y_test)
    gbdt_best = gbdt_model(x_train, x_test, y_train, y_test)

    for col in columns:
        _cb_score = _train(columns, cb_col, col, train_df, cb_model)
        _svm_score = _train(columns, svm_col, col, train_df, svm_model)
        _lr_score = _train(columns, lr_col, col, train_df, lr_model)
        _gbdt_score = _train(columns, gbdt_col, col, train_df, gbdt_model)
        if _cb_score >= cb_best:  # 删了这个特征分数提升或不变
            cb_best = _cb_score
            cb_col.append(col)
        if _svm_score >= svm_best:
            svm_best = _svm_score
            svm_col.append(col)
        if _lr_score >= lr_best:
            lr_best = _lr_score
            lr_col.append(col)
        if _gbdt_score >= gbdt_best:
            gbdt_best = _gbdt_score
            gbdt_col.append(col)
    print(cb_col)
    print(svm_col)
    print(lr_col)
    print(gbdt_col)



if __name__ == '__main__':
    train = pd.read_csv(analysis_path + 'all_features_train.csv')
    test = pd.read_csv(analysis_path + 'all_features_test.csv')

    test.rename(columns={"arpu_202005": "arpu_"}, inplace=True)
    train.drop(['city_name', 'county_name', 'label_y', 'label_x'], axis=1, inplace=True)
    test.drop(['city_name', 'county_name', 'phone_no_m'], axis=1, inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    test.loc[test['arpu_'] == '\\N', 'arpu_'] = 0

    elimination(train)

