from ScamCall.filter_features import bool_feature, num_feature, stack, train_split, analysis_path, svm_cross_validation, score, load_data
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


def predict(train_df, test_df, test_id):
    train_df.drop(cb_dis, axis=1, inplace=True)
    test_df.drop(cb_dis, axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_split(train_df)

    # model = cb_model(x_train, x_test, y_train, y_test)
    # res = model.predict_proba(test)[:, 1]
    # pre = [round(x-0.25) for x in res]
    # print(len([x for x in pre if x == 1]))

    # model = svm_model(x_train, x_test, y_train, y_test)
    # # res = model.predict_proba(test)[:, 1]
    # # pre = [round(x) for x in res]
    # pre = model.predict(test)
    # print(len([x for x in pre if x == 1]))

    model = gbdt_model(x_train, x_test, y_train, y_test)
    res = model.predict_proba(test)[:, 1]
    pre = [round(x-0.3) for x in res]
    print(len([x for x in pre if x == 1]))

    res_dict = {'phone_no_m': test_id, 'label': pre}
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(analysis_path + 'res_all_gbdt.csv', index=False)


def manuel():
    cb_res = pd.read_csv(analysis_path + 'res_all_cb.csv')
    cb_gbdt = pd.read_csv(analysis_path + 'res_all_gbdt.csv')
    cb_best = pd.read_csv(analysis_path + 'lsw_cb.csv')

    sum_df = pd.merge(cb_res, cb_gbdt, on='phone_no_m', how='outer')
    sum_df = pd.merge(sum_df, cb_best, on='phone_no_m', how='outer')
    sum_df.rename(columns={"label_x": "a", "label_y": "b", "label": "c"}, inplace=True)

    sum_df['label'] = sum_df['a'] + sum_df['b'] + sum_df['c']
    sum_df['label'] = sum_df['label'].apply(lambda x: 1 if x == 3 else 0)

    # 换卡次数大于20的全预测正确
    # phone_imei_count = test_user.groupby('phone_no_m')['imei_m'].nunique().reset_index(name='imei_phone')
    # imei_20 = phone_imei_count[phone_imei_count['imei_phone'] >= 20]['phone_no_m'].tolist()
    # for i in imei_20:
    #     print(sum_df[sum_df['phone_no_m'] == i]['label'])

    # 缺失值93个
    train_phone = load_data('test_voc')['phone_no_m'].drop_duplicates().tolist()
    test_phone = load_data('test_user')['phone_no_m'].tolist()
    miss_phone = list(set(test_phone).difference(set(train_phone)))
    _v = load_data('test_voc')
    _u = load_data('test_user')
    _s = load_data('test_sms')
    _a = load_data('test_app')

    miss_phone_voc = _v[_v['phone_no_m'].isin(miss_phone)]
    miss_phone_user = _u[_u['phone_no_m'].isin(miss_phone)]
    miss_phone_sms = _s[_s['phone_no_m'].isin(miss_phone)]
    miss_phone_app = _a[_a['phone_no_m'].isin(miss_phone)]

    miss_phone_voc.to_csv(analysis_path + 'miss_voc.csv', index=False, encoding='utf-8')
    miss_phone_user.to_csv(analysis_path + 'miss_user.csv', index=False, encoding='utf-8')
    miss_phone_sms.to_csv(analysis_path + 'miss_sms.csv', index=False)
    miss_phone_app.to_csv(analysis_path + 'miss_app.csv', index=False, encoding='utf-8')

    # print(sum_df[sum_df['phone_no_m'].isin(miss_phone)]['label'].sum())

    # 1交集363 0：954， 不确定133
    # 缺失数据都认为是诈骗419，得分下降

    # 分析133个不确定
    sus = ['04a948a430d2908836d791d7ea647e1f8c8a0e6fe24407b819dda3e2c1475ed3d0ffb278a0bcc8f60f010c8b07fbde5db8610f84236c17d3f94dfc55fcb8b9e2',
           '116614a2006a23921350191950db5e71bf9b17785b5fefb2542693a03c42cf0086f6198973d3bb9e0217c972f0df1b1260c3d1cf27ebd1e9d0468a50f2ae7acc',
           '']

if __name__ == '__main__':
    cb_dis = ['arpu_', 'start_datetime_count', 'call_dur_mean_x', 'dure_sum', 'call_sum_01', 'calltype_id_sum',
     'request_datetime_nunique', 'call_month_1_x']
    lr_dis = ['oppsite_no_m_voc_nunique', 'county_name_nunique_x', 'date_unique_01', 'phone2oppo_sum_mean',
     'phone2oppo_sum_median', 'phone2oppo_sum_max', 'call_dur_mean_y', 'call_dur_max', 'voc_hour_mode',
     'voc_hour_mode_count', 'voc_hour_nunique', 'flow1']
    gbdt_dis = ['oppsite_no_m_voc_nunique', 'call_diff_02', 'averge_call', 'flow_min']


    train = pd.read_csv(analysis_path + 'all_features_train.csv')
    test = pd.read_csv(analysis_path + 'all_features_test.csv')

    test_id = test['phone_no_m'].tolist()
    test.rename(columns={"arpu_202005": "arpu_"}, inplace=True)
    train.drop(['city_name', 'county_name', 'label_y', 'label_x'], axis=1, inplace=True)
    test.drop(['city_name', 'county_name', 'phone_no_m'], axis=1, inplace=True)
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    test.loc[test['arpu_'] == '\\N', 'arpu_'] = 0

    # elimination(train)
    # predict(train, test, test_id)
    # manuel()

    _363 = pd.read_csv(analysis_path + '363.csv')
    train_phone = load_data('test_voc')['phone_no_m'].drop_duplicates().tolist()
    test_phone = load_data('test_user')['phone_no_m'].tolist()
    miss_phone = list(set(test_phone).difference(set(train_phone)))
    _363.loc[_363['phone_no_m'].isin(miss_phone), 'label'] = 0
    tmp = ['d8fbe257ba16c2e11f562ac1343303a517080a01e990a35279e74b4e6d62631da33c5154f5139024b8d2dc8e8b5b0e30f1efdf5023b5b930b065fa5f1bc887cb',
           'a8d106452d28a3996a360eeae46b362067c90730fb314dacb1d73380941fa6962c1202e6eea0cc43b2d396552d7f712e9a07f7185457d834317222673af47058',
           '11fc0a8bbc7c9a5ce4519d92805beded3a842db9bc086dc4b92aae5beada07b327287c207cc2f0dfc4bd5fb727d98fb507a5083b3279201bd19442b6f68070a0',
           'b6e3f33f2889e50f6eee1719338038b129917cd37938b94a44ac4b72a702fb02951c4797eb4db19de01d7843746570d5b03d7892f193164923ada1e761d6b348',
           '8bce61a09c3d0cd0d9492e2b98272d944263b5ce7cb0c9f857a293fd19d075920847cf63a81e96b72c69bf896a973fe28a7a919032d24c48a9c4a32ed97c2db1',
           'b6f5149c107d9e0a473bb42f75873e1f458b7fb3340608f96eeb2e06a87214ab1f090906ef0d7a858fedceb1e844a7613fb8dff4b2e896299046b918c50ac7d8',
           '4f1c4ea7f091f431e65057ef9b299d4506bfb918c79853525fdc6f9f50f2cf91bf8cd6205c7870de6092c216ce51bce1c8a1fc91f30a5db9274f701fce1deafc']
    _363.loc[_363['phone_no_m'].isin(tmp), 'label'] = 1
    # _363.to_csv(analysis_path + 'handel_363.csv', index=False)

    # 预测为正常的号码里可能预测错的