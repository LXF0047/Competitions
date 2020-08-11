from ScamCall.analysis_best import load_data, recall, analysis_path, xff_path, train_split, cb_model, train_split_k
from ScamCall.analysis_best import after_week_recall
import threading
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import pickle
import prettytable as pt

res_path = '/home/lxf/data/analysis_files/res/'


def data(t):
    if t == 'test':
        df_app = load_data('test_app')
        df_voc = load_data('test_voc')
        same_call_count = pickle.load(open(xff_path + 'same_call_count_test.pkl', 'rb'))
        same_call_max = pickle.load(open(xff_path + 'same_call_max_test.pkl', 'rb'))
    elif t == 'train':
        df_app = load_data('train_app')
        df_voc = load_data('train_voc')
        same_call_count = pickle.load(open(xff_path + 'same_call_count_train.pkl', 'rb'))
        same_call_max = pickle.load(open(xff_path + 'same_call_max_train.pkl', 'rb'))
    elif t == 'p':
        df_app = pd.read_csv(analysis_path + 'app_positive.csv')
        df_voc = pd.read_csv(analysis_path + 'voc_positive.csv')
        same_call_count = pickle.load(open(xff_path + 'same_call_count_p.pkl', 'rb'))
        same_call_max = pickle.load(open(xff_path + 'same_call_max_p.pkl', 'rb'))
    elif t == 'n':
        df_app = pd.read_csv(analysis_path + 'app_negative.csv')
        df_voc = pd.read_csv(analysis_path + 'voc_negative.csv')
        same_call_count = pickle.load(open(xff_path + 'same_call_count_n.pkl', 'rb'))
        same_call_max = pickle.load(open(xff_path + 'same_call_max_n.pkl', 'rb'))
    else:
        return None, None, None, None

    return df_app, df_voc, same_call_count, same_call_max


def rule_analysis(t):
    # 加载数据
    df_app, df_voc, same_call_count, same_call_max = data(t)

    # 加载字典数据
    id2num, num2id = pickle.load(open(xff_path + 'new_dict.pkl', 'rb'))

    # 表格打印
    tb = pt.PrettyTable()
    tb.field_names = ["说明", "Hit", "Total"]

    # 流量使用月数为0或1
    month_p = df_app.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')
    flow_01 = month_p[month_p['month_count'] < 2]['phone_no_m'].to_list()
    tb.add_row(["流量使用月数为0或1", len(flow_01), df_app['phone_no_m'].drop_duplicates().shape[0]])
    # print('流量使用月数为0或1', len(flow_01))

    # 只使用了一个月流量
    flow_1 = month_p[month_p['month_count'] == 1]['phone_no_m'].to_list()
    tb.add_row(["只使用了一个月流量", len(flow_1), df_app['phone_no_m'].drop_duplicates().shape[0]])

    # 没使用过流量
    flow_0 = month_p[month_p['month_count'] == 0]['phone_no_m'].to_list()
    tb.add_row(["没使用过流量", len(flow_0), df_app['phone_no_m'].drop_duplicates().shape[0]])

    # 拨出+接听不同电话大于100
    # call_100 = df[df['phone_no_m'].isin(p_phone)].groupby('phone_no_m')[
    #     'opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_take_100 = df_voc.groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_take_100_phone = call_take_100[call_take_100['voc_nunique'] > 150]['phone_no_m'].to_list()
    tb.add_row(["拨出+接听不同电话大于150", len(call_take_100_phone), call_take_100.shape[0]])
    # print('拨出+接听不同电话大于150', len(call_take_100_phone))


    # 拨出100个电话以上
    call_100 = df_voc[df_voc['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_100_phone = call_100[call_100['voc_nunique'] > 200]['phone_no_m'].to_list()
    tb.add_row(["拨出200个电话以上", len(call_100_phone), call_100.shape[0]])
    # print('拨出100个电话以上', len(call_100_phone))

    # 主叫100次以上的电话回拨率
    phone_id_list = [num2id[x] for x in df_voc['phone_no_m'].drop_duplicates().tolist()]
    recall_count = recall(phone_id_list)  # {'phone': (回拨数, 拨出数)}
    n_ratio = [recall_count[x][0] / recall_count[x][1] for x in recall_count.keys() if
               recall_count[x][1] != 0 and recall_count[x][1] > 100]
    n_ra_0 = len([x for x in n_ratio if x == 0])
    tb.add_row(["主叫100次以上回拨率为0", n_ra_0, len(n_ratio)])
    # print('回拨率为0', n_ra_0, len(n_ratio))

    # 手机换卡次数大于10
    phone_imei_count = df_voc.groupby('phone_no_m')['imei_m'].nunique().reset_index(name='imei_phone')
    phone_imei_count_15 = phone_imei_count[phone_imei_count['imei_phone'] > 20]
    tb.add_row(["手机换卡次数大于20", phone_imei_count_15.shape[0], phone_imei_count.shape[0]])
    # print('手机换卡次数大于15', phone_imei_count_15.shape[0])

    # 主叫通话时长为0
    call_dur = df_voc[(df_voc['calltype_id'] == 1) & (df_voc['call_dur'] == 0)].groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='call_dur_0')
    # call_dur_all = df_voc[df_voc['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='call_dur_all')
    tb.add_row(["主叫通话时长为0", call_dur['call_dur_0'].shape[0], call_dur.shape[0]])
    # print('主叫通话时长为0的', call_dur['call_dur_0'].shape[0])

    # 给多少个相同的人打过电话
    same_c = len([x for x in same_call_count.values() if x > 200])
    same_cc = len([x for x in same_call_max.values() if x > 200])
    tb.add_row(["固定通话个数（打过10次以上的）大于200", same_c, len(same_call_count)])
    tb.add_row(["相同号码拨出电话的最多次数大于200", same_cc, len(same_call_max)])

    # 有通话记录的月份数
    df_voc['start_datetime'] = pd.to_datetime(df_voc['start_datetime'])
    # df_voc["year"] = df_voc['start_datetime'].dt.year
    # df_voc["hour"] = df_voc['start_datetime'].dt.hour
    # df_voc["day"] = df_voc['start_datetime'].dt.day
    df_voc["month"] = df_voc['start_datetime'].dt.month
    call_month = df_voc.groupby('phone_no_m')['month'].nunique().reset_index(name='call_month')
    call_month_1 = len([x for x in call_month['call_month'] if x == 1])
    tb.add_row(["通话月份只有1的（可能临时开卡）", call_month_1, call_month.shape[0]])

    # 通话月份数为1且流量月份数也为1
    call_month_1_phone = call_month[call_month['call_month'] == 1]['phone_no_m'].tolist()
    both_1 = len(list(set(call_month_1_phone).intersection(set(flow_1))))
    tb.add_row(["通话月份只为1且流量月数也为1", both_1, call_month.shape[0]])

    # 拨出不同号码数大于100
    more_100 = df_voc.groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='tmp')
    more = more_100[more_100['tmp'] > 200].shape[0]
    tb.add_row(["拨出不同号码数大于100", more, more_100.shape[0]])

    # 一星期后任然有联系的电话数

    print(tb)


def bool_feature(t):
    print('数据处理中', end='')
    for i in range(3):
        print('.', end='')
        time.sleep(0.2)

    # 加载数据
    df_app, df_voc, same_call_count, same_call_max = data(t)

    # merge list
    merge_list = []

    # label
    if t == 'train':
        label = load_data('train_user')[['phone_no_m', 'label']]
        merge_list.append(label)

    # 流量使用月数为0或1
    month_p = df_app.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')
    month_p.loc[month_p['month_count'] < 2, 'flow_01'] = 1
    month_p.loc[month_p['month_count'] >= 2, 'flow_01'] = 0
    merge_list.append(month_p[['phone_no_m', 'flow_01']])

    # 只使用了一个月流量
    flow_1 = month_p[month_p['month_count'] == 1]['phone_no_m'].to_list()
    month_p.loc[month_p['month_count'] == 1, 'flow1'] = 1
    month_p.loc[month_p['month_count'] != 1, 'flow1'] = 0
    merge_list.append(month_p[['phone_no_m', 'flow1']])

    # 没使用过流量
    month_p.loc[month_p['month_count'] == 0, 'flow0'] = 1
    month_p.loc[month_p['month_count'] != 0, 'flow0'] = 0
    merge_list.append(month_p[['phone_no_m', 'flow0']])

    # 拨出+接听不同电话大于200
    call_take_100 = df_voc.groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_take_100.loc[call_take_100['voc_nunique'] >= 200, 'call_take_100'] = 1
    call_take_100.loc[call_take_100['voc_nunique'] < 200, 'call_take_100'] = 0
    merge_list.append(call_take_100[['phone_no_m', 'call_take_100']])

    # # 主叫100次以上的电话回拨率
    # phone_id_list = [num2id[x] for x in df_voc['phone_no_m'].drop_duplicates().tolist()]
    # recall_count = recall(phone_id_list)  # {'phone': (回拨数, 拨出数)}
    # n_ratio = [recall_count[x][0] / recall_count[x][1] for x in recall_count.keys() if
    #            recall_count[x][1] != 0 and recall_count[x][1] > 100]
    # n_ra_0 = len([x for x in n_ratio if x == 0])
    # # print('回拨率为0', n_ra_0, len(n_ratio))

    # 手机换卡次数大于20
    phone_imei_count = df_voc.groupby('phone_no_m')['imei_m'].nunique().reset_index(name='imei_phone')
    phone_imei_count.loc[phone_imei_count['imei_phone'] >= 20, 'phone_imei_count_20'] = 1
    phone_imei_count.loc[phone_imei_count['imei_phone'] < 20, 'phone_imei_count_20'] = 0
    merge_list.append(phone_imei_count[['phone_no_m', 'phone_imei_count_20']])

    # 有通话记录的月份数
    df_voc['start_datetime'] = pd.to_datetime(df_voc['start_datetime'])
    df_voc["month"] = df_voc['start_datetime'].dt.month
    call_month = df_voc.groupby('phone_no_m')['month'].nunique().reset_index(name='call_month')
    call_month.loc[call_month['call_month'] == 1, 'call_month_1'] = 1
    call_month.loc[call_month['call_month'] != 1, 'call_month_1'] = 0
    merge_list.append(call_month[['phone_no_m', 'call_month_1']])

    # 通话月份数为1且流量月份数也为1
    call_month_1_phone = call_month[call_month['call_month'] == 1]['phone_no_m'].tolist()
    both_1 = list(set(call_month_1_phone).intersection(set(flow_1)))
    call_month.loc[call_month['phone_no_m'].isin(both_1), 'call_flow_1'] = 1
    call_month['call_flow_1'].fillna(0, inplace=True)

    # merge
    new_df = pd.merge(merge_list[0], merge_list[1], on='phone_no_m', how='outer')
    for i in range(2, len(merge_list)):
        new_df = pd.merge(new_df, merge_list[i], on='phone_no_m', how='outer')
    # print(new_df.shape[0])
    return new_df


def num_feature(t):
    # 加载数据
    df_app, df_voc, same_call_count, same_call_max = data(t)

    # merge list
    merge_list = []

    # label
    if t == 'train':
        label = load_data('train_user')[['phone_no_m', 'label']]
        merge_list.append(label)

    # 流量使用月数
    month_p = df_app.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')
    merge_list.append(month_p[['phone_no_m', 'month_count']])

    # 拨出+接听不同电话
    call_take_100 = df_voc.groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='voc_nunique')
    merge_list.append(call_take_100[['phone_no_m', 'voc_nunique']])

    # 手机换卡次数
    phone_imei_count = df_voc.groupby('phone_no_m')['imei_m'].nunique().reset_index(name='imei_phone')
    merge_list.append(phone_imei_count[['phone_no_m', 'imei_phone']])

    # 有通话记录的月份数
    df_voc['start_datetime'] = pd.to_datetime(df_voc['start_datetime'])
    df_voc["month"] = df_voc['start_datetime'].dt.month
    call_month = df_voc.groupby('phone_no_m')['month'].nunique().reset_index(name='call_month')
    call_month.loc[call_month['call_month'] == 1, 'call_month_1'] = 1
    call_month.loc[call_month['call_month'] != 1, 'call_month_1'] = 0
    merge_list.append(call_month[['phone_no_m', 'call_month_1']])

    # 通话月份数为1且流量月份数也为1
    flow_1 = month_p[month_p['month_count'] == 1]['phone_no_m'].to_list()
    call_month_1_phone = call_month[call_month['call_month'] == 1]['phone_no_m'].tolist()
    both_1 = list(set(call_month_1_phone).intersection(set(flow_1)))
    call_month.loc[call_month['phone_no_m'].isin(both_1), 'call_flow_1'] = 1
    call_month['call_flow_1'].fillna(0, inplace=True)
    merge_list.append(call_month[['phone_no_m', 'call_flow_1']])

    # merge
    new_df = pd.merge(merge_list[0], merge_list[1], on='phone_no_m', how='outer')
    for i in range(2, len(merge_list)):
        new_df = pd.merge(new_df, merge_list[i], on='phone_no_m', how='outer')

    return new_df


def same_call(df, t):
    from tqdm import tqdm
    phone_list = df['phone_no_m'].drop_duplicates().tolist()
    same_call_count = {}
    same_call_max = {}
    for phone in tqdm(phone_list):
        v_c = df[(df['phone_no_m'] == phone) & (df['calltype_id'] == 1)]['opposite_no_m'].value_counts().reset_index(name='v_c')
        same_call_count[phone] = v_c[v_c['v_c'] > 10]['v_c'].count()
        same_call_max[phone] = v_c[v_c['v_c'] > 10]['v_c'].max()
    pickle.dump(same_call_count, open(xff_path + 'same_call_count_%s.pkl' % t, 'wb'))
    pickle.dump(same_call_max, open(xff_path + 'same_call_max_%s.pkl' % t, 'wb'))


def svm_model(t):
    # 0.75581
    df = bool_feature(t)
    df.fillna(0, inplace=True)
    x_train, x_test, y_train, y_test = train_split(df)
    svm_m = svm_cross_validation(x_train, y_train)
    pre = svm_m.predict(x_test)
    score(pre, y_test)
    return svm_m


def svm_cross_validation(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def lr_model(t):
    # 0.7515
    df = bool_feature(t)
    df.fillna(0, inplace=True)
    x_train, x_test, y_train, y_test = train_split(df)
    lr = LogisticRegression()
    clf = lr.fit(x_train, y_train)
    pre = clf.predict(x_test)
    score(pre, y_test)
    return clf


def cb_m():
    train = pd.read_csv(analysis_path + 'test_train.csv')
    test = pd.read_csv(analysis_path + 'test_test.csv')
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    test.loc[test['arpu_202005'] == '\\N', 'arpu_202005'] = 0

    # 处理列
    test_id = test['phone_no_m'].tolist()
    train.drop(['county_name', 'city_name'], axis=1, inplace=True)
    test.drop(['phone_no_m'], axis=1, inplace=True)
    train_col = train.columns.to_list()
    train_col.remove('label')
    train_col.remove('phone_no_m')
    test.rename(columns={"arpu_202005": "arpu_"}, inplace=True)
    test = test[train_col]

    c_list = []  # 'city_name', 'county_name', 'calltype_id'
    x_train, x_test, y_train, y_test = train_split(train)

    # 交叉验证
    # kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2020)  # 3折交叉
    # purchase_result = 0  # 最终结果
    # data_k, label_k = train_split_k(train)
    # for train_index, test_index in kf.split(data_k, label_k):
    #     X_train_p, y_train_p = data_k.iloc[train_index], label_k.iloc[train_index]
    #     X_valid_p, y_valid_p = data_k.iloc[test_index], label_k.iloc[test_index]
    #     purchase_result += cb_model(X_train_p, X_valid_p, y_train_p, y_valid_p, test, c_list) / 3.0
    # res_dict = {'phone_no_m': test_id, 'label': [value-0.1 for value in purchase_result]}  # round(value-0.3)
    # res_df = pd.DataFrame(res_dict)
    # res_df.to_csv(res_path + 'res_cb_3-01.csv', index=False)

    # 开始训练
    res = cb_model(x_train, x_test, y_train, y_test, test, c_list)
    predictions = [round(value - 0.2) for value in res]
    res_dict = {'phone_no_m': test_id, 'label': predictions}
    res_df = pd.DataFrame(res_dict)

    print(res_df['label'].sum())

    res_df.to_csv(res_path + 'res_cb.csv', index=False)


def score(pre, true):
    from sklearn.metrics import f1_score
    score = f1_score(true, pre, average='macro')
    print('F1: %s' % score)


def lsw():
    from catboost import CatBoostClassifier
    train = pd.read_csv(analysis_path + 'train_result.csv')
    test = pd.read_csv(analysis_path + 'test_result.csv')

    # merge svm lr results
    train = stack('train', train)
    test = stack('test', test)

    test_id = test['phone_no_m']
    test.loc[test['arpu'] == '\\N', 'arpu'] = 0
    test.drop(['phone_no_m'], axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_split(train)

    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, loss_function='Logloss',
                               logging_level='Verbose', eval_metric='F1')
    model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=200, silent=True)
    res = model.predict_proba(test)[:, 1]
    # predictions = model.predict(test)
    res_dict = {'phone_no_m': test_id, 'label': [round(value-0.2) for value in res]}
    res_df = pd.DataFrame(res_dict)
    print(res_df['label'].sum())

    res_df.to_csv(analysis_path + 'lsw_cb.csv', index=False)
    return res_df


def stack(t, init_df):
    lr = lr_model('train')  # 训练模型
    svm = svm_model('train')
    if t == 'train':
        train = pd.read_csv(analysis_path + 'train_result.csv')
        df = bool_feature('train')
        _id = df['phone_no_m']
        missing_list = list(set(_id.tolist()).difference(set(df['phone_no_m'].tolist())))
        df.drop(['phone_no_m', 'label'], axis=1, inplace=True)
    else:
        test = pd.read_csv(analysis_path + 'test_result.csv')
        df = bool_feature('test')
        _id = df['phone_no_m']
        missing_list = list(set(_id.tolist()).difference(set(df['phone_no_m'].tolist())))
        df.drop(['phone_no_m'], axis=1, inplace=True)

    df.fillna(0, inplace=True)

    lr_res = lr.predict(df)
    svm_res = svm.predict(df)

    dict_lr = {'phone_no_m': _id, 'lr_res': lr_res}
    dict_svm = {'phone_no_m': _id, 'lr_res': svm_res}

    # 补充缺失电话label，认为都是诈骗
    for phone in missing_list:
        if phone not in dict_lr:
            dict_lr[phone] = 1
        if phone not in dict_svm:
            dict_svm[phone] = 1

    df_lr = pd.DataFrame(dict_lr)
    df_svm = pd.DataFrame(dict_svm)

    tmp = pd.merge(init_df, df_lr, on='phone_no_m', how='outer')
    res_df = pd.merge(tmp, df_svm, on='phone_no_m', how='outer')

    return res_df
    # return None


if __name__ == '__main__':
    # svm_model('test')
    # lr_model('test')
    # cb_m()
    lsw()  # 567
    # bool_feature('test')
    '''
    >>> 诈骗
    +---------------------------------------+------+-------+
    |                  说明                 | Hit  | Total |
    +---------------------------------------+------+-------+
    |           流量使用月数为0或1            | 1196 |  1962 |
    |           只使用了一个月流量            | 690  |  1962 |
    |              没使用过流量              | 506  |  1962 |
    |        拨出+接听不同电话大于150          | 888  |  1892 |
    |           拨出200个电话以上            | 630  |  1876 |
    |         主叫100次以上回拨率为0          | 498  |  1038 |
    |           手机换卡次数大于20            |  29  |  1892 |
    |            主叫通话时长为0              |  7   |   7   |
    | 固定通话个数（打过10次以上的）大于200      |  1   |  1892 |
    |   相同号码拨出电话的最多次数大于200        |  78  |  1892 |
    |    通话月份只有1的（可能临时开卡）        | 742  |  1892 |
    |      通话月份只为1且流量月数也为1         | 456  |  1892 |
    +---------------------------------------+------+-------+
    
    >>> 正常
    +---------------------------------------+-----+-------+
    |                  说明                 | Hit | Total |
    +---------------------------------------+-----+-------+
    |           流量使用月数为0或1             | 493 |  4144 |
    |           只使用了一个月流量             | 187 |  4144 |
    |              没使用过流量               | 306 |  4144 |
    |        拨出+接听不同电话大于150           | 919 |  4133 |
    |           拨出200个电话以上              | 244 |  4113 |
    |         主叫100次以上回拨率为0            |  51 |  774  |
    |           手机换卡次数大于20              |  1  |  4133 |
    |            主叫通话时长为0                |  0  |   0   |
    | 固定通话个数（打过10次以上的）大于200        |  0  |  4133 |
    |   相同号码拨出电话的最多次数大于200          | 498 |  4133 |
    |    通话月份只有1的（可能临时开卡）           | 111 |  4133 |
    |      通话月份只为1且流量月数也为1           |  80 |  4133 |
    +---------------------------------------+-----+-------+
    '''


    # rule_analysis('n')
    # print('*' * 100)
    # rule_analysis('p')
