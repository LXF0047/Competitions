from ScamCall.filter_features import bool_feature, num_feature, stack, train_split, analysis_path, svm_cross_validation, score
from ScamCall.baseline089 import feats
from ScamCall.analysis_all_feature import main
from ScamCall.analysis_best import eda
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
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
    model = SVC()
    model.fit(x_train, y_train)
    pre = model.predict(x_test)
    s = score(pre, y_test)
    print(s)
    return model


def lr_model(x_train, x_test, y_train, y_test):
    lr = LogisticRegression()
    clf = lr.fit(x_train, y_train)
    pre = clf.predict(x_test)
    s = score(pre, y_test)
    return clf


def gbdt_model(x_train, x_test, y_train, y_test):
    gbm0 = GradientBoostingClassifier(random_state=2020)
    gbm0.fit(x_train, y_train)
    pre = gbm0.predict(x_test)
    s = score(pre, y_test)
    return gbm0


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

    return model


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


def predict(train_df, test_df):
    import numpy as np
    train_df.drop(svm_dis, axis=1, inplace=True)
    test_df.drop(svm_dis, axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_split(train_df)
    svm_ = svm_model(x_train, x_test, y_train, y_test)
    res = svm_.predict(x_test)
    print(len([x for x in res if x == 1]), np.sum(y_test))



if __name__ == '__main__':
    cb_dis = ['arpu_', 'start_datetime_count', 'call_dur_mean_x', 'dure_sum', 'call_sum_01', 'calltype_id_sum',
     'request_datetime_nunique', 'call_month_1_x']
    svm_dis = ['idcard_cnt', 'arpu_', 'oppsite_no_m_voc_nunique', 'call_diff_01', 'call_diff_02', 'start_datetime_count',
     'date_unique', 'call_dur_mean_x', 'city_name_nunique_x', 'county_name_nunique_x', 'imei_m_nunique', 'ratio',
     'dure_sum', 'call_sum', 'call_sum_01', 'call_sum_02', 'date_unique_01', 'date_unique_02', 'call_day_max',
     'call_day_01_max', 'call_day_02_max', 'averge_call', 'averge_call_01', 'dure_std', 'dure_mean', 'phone_count_max',
     'imei_county_mean', 'averge_call_02', 'opposite_no_m_nunique', 'calltype_id_sum', 'request_datetime_nunique',
     'busi_name_nunique', 'flow_sum_x', 'app_month', 'opposite_count', 'opposite_unique', 'voccalltype1', 'imeis',
     'voc_calltype1', 'city_name_call', 'county_name_call', 'phone2opposite_mean', 'phone2opposite_median',
     'phone2opposite_max', 'phone2oppo_sum_mean', 'phone2oppo_sum_median', 'phone2oppo_sum_max', 'call_dur_mean_y',
     'call_dur_median', 'call_dur_max', 'call_dur_min', 'city_name_nunique_y', 'county_name_nunique_y',
     'calltype_id_unique', 'voc_hour_mode', 'voc_hour_mode_count', 'voc_hour_nunique', 'voc_day_mode',
     'voc_day_mode_count', 'voc_day_nunique', 'busi_count', 'flow_mean', 'flow_median', 'flow_min', 'flow_max',
     'flow_var', 'flow_sum_y', 'month_ids', 'flow_month', 'flow_01', 'flow1', 'flow0', 'call_take_100',
     'phone_imei_count_20', 'call_month_1_x', 'voc_nunique', 'call_month_1_y']
    lr_dis = ['oppsite_no_m_voc_nunique', 'county_name_nunique_x', 'date_unique_01', 'phone2oppo_sum_mean',
     'phone2oppo_sum_median', 'phone2oppo_sum_max', 'call_dur_mean_y', 'call_dur_max', 'voc_hour_mode',
     'voc_hour_mode_count', 'voc_hour_nunique', 'flow1']
    gbdt_dis = ['oppsite_no_m_voc_nunique', 'call_diff_02', 'averge_call', 'flow_min']


    train = pd.read_csv(analysis_path + 'all_features_train.csv')
    test = pd.read_csv(analysis_path + 'all_features_test.csv')

    test.rename(columns={"arpu_202005": "arpu_"}, inplace=True)
    train.drop(['city_name', 'county_name', 'label_y', 'label_x'], axis=1, inplace=True)
    test.drop(['city_name', 'county_name', 'phone_no_m'], axis=1, inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    test.loc[test['arpu_'] == '\\N', 'arpu_'] = 0

    # elimination(train)
    predict(train, test)

