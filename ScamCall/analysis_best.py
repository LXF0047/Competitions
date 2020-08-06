import pandas as pd
from tqdm import tqdm
import operator
import pandas_profiling
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import roc_auc_score
import seaborn as sns
import threading
from matplotlib import pyplot as plt
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from multiprocessing.dummy import Pool as ThreadPool
from pathos.multiprocessing import ProcessingPool as newPool
import multiprocessing

g_res_set = {}
xff_path = '/home/lxf/data/xff/'
analysis_path = '/home/lxf/data/analysis_files/'

'''加载数据集'''
def load_data(file):
    path = '/home/lxf/data/'
    test_path = '/home/lxf/data/'
    # 手机应用使用情况  ['phone_no_m', 'busi_name', 'flow', 'month_id']  号码 APP名称 使用流量 月分区
    if file == 'train_app':
        return pd.read_csv(path + 'train_app.csv')
    # 短信情况  ['phone_no_m', 'opposite_no_m', 'calltype_id', 'request_datetime']  发出号码  接收号码  短信类型  发送日期
    elif file == 'train_sms':
        return pd.read_csv(path + 'train_sms.csv')
    # 基础信息  ['phone_no_m', 'city_name', 'county_name', 'idcard_cnt', 'arpu_201908',    拨出号码  开户市 开户区县 名下号码量
    #          'arpu_201909', 'arpu_201910', 'arpu_201911', 'arpu_201912', 'arpu_202001',   月消费值 label
    #          'arpu_202002', 'arpu_202003', 'label']
    elif file == 'train_user':
        return pd.read_csv(path + 'train_user.csv')
    # 语音情况  ['phone_no_m', 'opposite_no_m', 'calltype_id', 'start_datetime',
    #                           'call_dur', 'city_name', 'county_name', 'imei_m']
    elif file == 'train_voc':
        return pd.read_csv(path + 'train_voc.csv', low_memory=False)
    # 测试
    elif file == 'test_app':
        return pd.read_csv(test_path + 'test_app.csv')
    elif file == 'test_sms':
        return pd.read_csv(test_path + 'test_sms.csv')
    elif file == 'test_user':
        return pd.read_csv(test_path + 'test_user.csv')
    elif file == 'test_voc':
        return pd.read_csv(test_path + 'test_voc.csv')
    else:
        return None


'''处理数据里并存入新的csv'''
def handel_dataset(time, c):
    _app = load_data('%s_app' % c)
    _sms = load_data('%s_sms' % c)
    _user = load_data('%s_user' % c)
    _voc = load_data('%s_voc' % c)

    voc_df = handel_voc(_voc)
    sms_df = handel_sms(_sms)
    app_df = handel_app(_app)
    if c == 'train':
        user_df = handel_user(_user)
    else:
        user_df = _user

    # data_set = user_df
    data_set = pd.merge(user_df, voc_df, on='phone_no_m', how='outer')
    data_set = pd.merge(data_set, sms_df, on='phone_no_m', how='outer')
    data_set = pd.merge(data_set, app_df, on='phone_no_m', how='outer')

    data_set.to_csv('./resource/train_test_res/%s/%s.csv' % (time, c), index=False)


def handel_user(df):
    # 基本信息处理
    df['arpu_'] = (df['arpu_201908'] + df['arpu_201909'] + df['arpu_201910'] + df['arpu_201911'] +
                   df['arpu_201912'] + df['arpu_202001'] + df['arpu_202002'] + df['arpu_202003']) / 8
    df['arpu_'].fillna(0, inplace=True)

    df.drop(['arpu_201908', 'arpu_201909', 'arpu_201910', 'arpu_201911', 'arpu_201912', 'arpu_202001',
              'arpu_202002', 'arpu_202003'], axis=1, inplace=True)


    return df


def handel_sms(df):
    # 短信信息处理
    df['calltype_id'] = df['calltype_id'] - 1

    opposite_no_m_nunique = df.groupby('phone_no_m', sort=False)['opposite_no_m'].nunique().reset_index(
        name='opposite_no_m_nunique')
    # 短信上行总和（回短信）
    calltype_id_sum = df.groupby('phone_no_m', sort=False)['calltype_id'].sum().reset_index(name='calltype_id_sum')
    request_datetime_nunique = df.groupby('phone_no_m', sort=False)['request_datetime'].nunique().reset_index(
        name='request_datetime_nunique')

    new_df = pd.merge(opposite_no_m_nunique, calltype_id_sum, on='phone_no_m', how='outer')
    new_df = pd.merge(new_df, request_datetime_nunique, on='phone_no_m', how='outer')

    # new_df.to_csv('./resource/sms_%s_1.csv' % c, index=False)
    return new_df


def handel_app(df):
    # app信息处理
    # busi_name_nunique = df.groupby('phone_no_m', sort=False)['busi_name'].nunique().reset_index(
    #     name='busi_name_nunique')
    # flow_sum = df.groupby('phone_no_m', sort=False)['flow'].sum().reset_index(name='flow_sum')
    #
    # new_df = pd.merge(busi_name_nunique, flow_sum, on='phone_no_m', how='outer')
    # 可以尝试增加每月流量变化幅度的特征
    app_month = df.groupby('phone_no_m')['month_id'].nunique().reset_index(name='app_month')

    # new_df.to_csv('./resource/app_%s_1.csv' % c, index=False)
    return app_month


def handel_voc(df):
    # 通话信息处理

    # 拆分时间列为日期和时间
    df['date'] = df['start_datetime'].map(lambda x: x.split(' ')[0])
    df['time'] = df['start_datetime'].map(lambda x: x.split(' ')[1])
    df['date'] = df['start_datetime'].map(lambda x: x.split(' ')[0])
    df['time'] = df['start_datetime'].map(lambda x: x.split(' ')[1])

    # 号码一共有多少通话记录  线下降
    call_sum = df.groupby('phone_no_m')['opposite_no_m'].count().reset_index(name='call_sum')
    # 号码主叫总数  线下降
    call_sum_01 = df[df['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].count().reset_index(name='call_sum_01')
    # 主叫不同号码数  增
    call_diff_01 = df[df['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='call_diff_01')
    # 号码通话的不同号码数 base
    oppsite_no_m_voc_nunique = df.groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(
        name='oppsite_no_m_voc_nunique')
    # 号码被叫总数  降
    call_sum_02 = df[df['calltype_id'] == 2].groupby('phone_no_m')['opposite_no_m'].count().reset_index(name='call_sum_02')
    # 号码被多少个不同号码叫 增
    call_diff_02 = df[df['calltype_id'] == 2].groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='call_diff_02')


    # 一个手机号的通话天数  奇怪的增长
    start_datetime_count = df.groupby('phone_no_m')['start_datetime'].count().reset_index(name='start_datetime_count')
    # 一个手机号的平均通话时长
    call_dur_mean = df.groupby('phone_no_m')['call_dur'].mean().reset_index(name='call_dur_mean')
    # 手机号在几个不同城市打过电话
    city_name_nunique = df.groupby('phone_no_m')['city_name'].nunique().reset_index(name='city_name_nunique')
    county_name_nunique = df.groupby('phone_no_m')['county_name'].nunique().reset_index(name='county_name_nunique')
    # 一个号码用了几个手机
    imei_m_nunique = df.groupby('phone_no_m')['imei_m'].nunique().reset_index(name='imei_m_nunique')


    # 一个号码通话的不同日期数  增
    date_unique = df.groupby('phone_no_m')['date'].nunique().reset_index(name='date_unique')
    # 号码主叫不同日期数  降
    date_unique_01 = df[df['calltype_id'] == 1].groupby('phone_no_m')['date'].nunique().reset_index(name='date_unique_01')
    # 号码被叫不同日期数  降
    date_unique_02 = df[df['calltype_id'] == 2].groupby('phone_no_m')['date'].nunique().reset_index(
        name='date_unique_02')
    # 一个号码一天最多通话几次  降
    call_day_max = df.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].max().reset_index(name='call_day_max')
    # 一个号码一天最多呼出几次  降
    call_day_01_max = df[df['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
        'phone_no_m')['nn'].max().reset_index(name='call_day_01_max')
    # 一个号码一天最多呼入几次  降
    call_day_02_max = df[df['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(
        name='nn').groupby('phone_no_m')['nn'].max().reset_index(name='call_day_02_max')
    # 号码平均每天通话次数  降
    # averge_call = df.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].mean().reset_index(name='averge_call')
    # 号码平均每天呼出次数  略降
    averge_call_01 = df[df['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
        'phone_no_m')['nn'].mean().reset_index(name='averge_call_01')
    # 号码平均呼入次数  降
    # averge_call_02 = df[df['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
    #     'phone_no_m')['nn'].mean().reset_index(name='averge_call_02')
    # 通话总时长  略降
    dure_sum = df.groupby('phone_no_m')['call_dur'].sum().reset_index(name='dure_sum')
    # 通话时长标准差
    dure_std = df.groupby('phone_no_m')['call_dur'].std().reset_index(name='dure_std')
    # 每次通话平均时长
    dure_mean = df.groupby('phone_no_m')['call_dur'].mean().reset_index(name='dure_mean')

    # 呼出时长小于5s的概率
    less5sec = df[(df['calltype_id'] == 1) & (df['call_dur'] < 5)].groupby('phone_no_m')['calltype_id'].count().reset_index(name='tmp1')
    total = df[df['calltype_id'] == 1].groupby('phone_no_m')['calltype_id'].count().reset_index(name='tmp2')
    tmp_df = pd.merge(less5sec, total, on='phone_no_m', how='outer')
    tmp_df['ratio'] = tmp_df['tmp1']/tmp_df['tmp2']
    tmp_ratio = tmp_df[['phone_no_m', 'ratio']]

    # 一个号码所有通话中重复拨出的号码

    # 递归得到特征
    # 每天最多用几个不同的手机'r_group_date_imei_m_value_counts_max', 线上降
    phone_count_max = df.groupby(['phone_no_m', 'date'])['imei_m'].value_counts().reset_index(name='tmp').groupby('phone_no_m')[
                            'tmp'].max().reset_index(name='phone_count_max')[['phone_no_m', 'phone_count_max']]
    # 一个号码用过的手机的开卡区县数均值
    # 'r_group_imei_m_county_name_count_mean',
    imei_county_mean = \
    df.groupby(['phone_no_m', 'imei_m'])['county_name'].value_counts().reset_index(name='tmp').groupby('phone_no_m')[
        'tmp'].max().reset_index(name='imei_county_mean')[['phone_no_m', 'imei_county_mean']]


    # 固定通话统计
    # tmp = df.groupby(["phone_no_m", "opposite_no_m"])["call_dur"].agg(count="count", sum="sum")
    # phone2opposite1 = tmp.groupby("phone_no_m")["count"].agg(phone2opposite_mean="mean", phone2opposite_median="median",
    #                                                         phone2opposite_max="max")
    #
    # phone2opposite2 = tmp.groupby("phone_no_m")["sum"].agg(phone2oppo_sum_mean="mean", phone2oppo_sum_median="median",
    #                                                       phone2oppo_sum_max="max")


    # 一个手机用过的电话卡数量
    # imei_phones = df.groupby('imei_m')['phone_no_m'].nunique().reset_index(name='imei_phones_count')
    # tmp_dic = dict(zip(imei_phones['imei_m'].tolist(), imei_phones['imei_phones_count'].tolsit()))
    # imei_phoens_c = df.groupby('phone_no_m')['imei_m'].value_counts().reset_index(name='tmp')




    m_list = [oppsite_no_m_voc_nunique, call_diff_01, call_diff_02, start_datetime_count, date_unique,
              call_dur_mean, city_name_nunique, county_name_nunique, imei_m_nunique]
    m_list_other = [averge_call_01, tmp_ratio, dure_sum]  #, phone_count_max
    total = m_list+m_list_other
    new_df = pd.merge(total[0], total[1], on='phone_no_m', how='outer')
    for i in range(2, len(total)):
        new_df = pd.merge(new_df, total[i], on='phone_no_m', how='outer')

    # new_df.to_csv('./resource/voc_%s_1.csv' % c, index=False)
    return new_df


def get_imei_phone(df):
    tmp1 = df.groupby('imei_m')['phone_no_m'].nunique().reset_index(name='tmp')
    dic1 = dict(zip(tmp1['imei_m'].tolist(), tmp1['tmp'].tolist()))
    phone_list = df['phone_no_m'].drop_duplicates().tolist()
    res_dict = {}
    for phone in tqdm(phone_list):
        imei = df[df['phone_no_m'] == phone]['imei_m'].tolist()
        imei_phone = 0
        for i in imei:
            if dic1[i] == 1:
                continue
            else:
                imei_phone += dic1[i]
        res_dict[phone] = imei_phone/len(imei)
    # print(res_dict.values())
    pickle.dump(res_dict, open('resource/xff/imei_phone_4.pkl', 'wb'))


def eda(df, filename='eda'):
    output_path = './resource/%s.html' % filename
    profile = pandas_profiling.ProfileReport(df, minimal=False)
    profile.to_file(output_file=output_path)


def train_split(df):
    train_label = df['label']
    train_data = df.drop(['phone_no_m','label'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3, random_state=2020)
    return x_train, x_test, y_train, y_test


def train_split_k(df):
    train_label = df['label']
    train_data = df.drop(['phone_no_m', 'label'], axis=1)
    return train_data, train_label


def cb_model(X_t, X_v, y_t, y_v, test, c_list):
    print('[INFO] Catboost Model starts training ...')
    category = []  #
    for index, value in enumerate(X_t.columns):
        if (value in c_list):
            category.append(index)
            continue
    str_int = list(np.where(X_t.dtypes != np.float)[0])
    # category = [0, 1]
    model = CatBoostClassifier(iterations=2000, learning_rate=0.05, cat_features=category, loss_function='Logloss',
                               logging_level='Verbose', eval_metric='F1')
    model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=200, silent=True)
    res = model.predict_proba(test)[:, 1]
    # res = model.predict(test)
    importance=model.get_feature_importance(prettified=True)  # 显示特征重要程度
    # print(importance)
    return res


def plot_analysis():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    # ***USER***
    # 开户区县诈骗号是否有影响，诈骗号有没有可能只从小地方开卡，或者没有开卡信息
    # 诈骗电话的月消费缺失值比较多，可能不是连续交费，但是如何利用？
    # ***VOC***
    # 被叫号码总数5015430，去重后1259878
    # 寻找号码之间的通话关系
    # 拨出不同号码的个数
    pass
    voc_n = pd.read_csv('./resource/analysis_files/voc_negative.csv')
    voc_p = pd.read_csv('./resource/analysis_files/voc_positive.csv')
    voc_n['date'] = voc_n['start_datetime'].map(lambda x: x.split(' ')[0])
    voc_n['time'] = voc_n['start_datetime'].map(lambda x: x.split(' ')[1])
    voc_p['date'] = voc_p['start_datetime'].map(lambda x: x.split(' ')[0])
    voc_p['time'] = voc_p['start_datetime'].map(lambda x: x.split(' ')[1])
    # 收费号码的通话总数  降
    # n = voc_n.groupby('phone_no_m')['opposite_no_m'].count()
    # p = voc_p.groupby('phone_no_m')['opposite_no_m'].count()
    # 收费号码通话不同号码数  已有
    # n = voc_n.groupby('phone_no_m')['opposite_no_m'].nunique()
    # p = voc_p.groupby('phone_no_m')['opposite_no_m'].nunique()
    # 收费号码主叫总数  降
    # n = voc_n[voc_n['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].count()
    # p = voc_p[voc_p['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].count()
    # 收费号码主叫不同号码数  增
    # n = voc_n[voc_n['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].nunique()
    # p = voc_p[voc_p['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].nunique()
    # 收费号码被叫总数  降
    # n = voc_n[voc_n['calltype_id'] == 2].groupby('phone_no_m')['opposite_no_m'].count()
    # p = voc_p[voc_p['calltype_id'] == 2].groupby('phone_no_m')['opposite_no_m'].count()
    # 收费号码被叫不同号码数  增
    # n = voc_n[voc_n['calltype_id'] == 2].groupby('phone_no_m')['opposite_no_m'].nunique()
    # p = voc_p[voc_p['calltype_id'] == 2].groupby('phone_no_m')['opposite_no_m'].nunique()

    # 收费号码不同通话日期数  增
    # n = voc_n.groupby('phone_no_m')['date'].nunique()
    # p = voc_p.groupby('phone_no_m')['date'].nunique()
    # 收费号码主叫时通话不同日期数  降
    # n = voc_n[voc_n['calltype_id'] == 1].groupby('phone_no_m')['date'].nunique()
    # p = voc_p[voc_p['calltype_id'] == 1].groupby('phone_no_m')['date'].nunique()
    # 收费号码被叫时不同通话日期数  降
    # n = voc_n[voc_n['calltype_id'] == 2].groupby('phone_no_m')['date'].nunique()
    # p = voc_p[voc_p['calltype_id'] == 2].groupby('phone_no_m')['date'].nunique()
    # 收费号码一天最多通话几次  降
    # n = voc_n.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].max()
    # p = voc_p.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].max()
    # 收费号码一天最多呼出几次  降
    # n = voc_n[voc_n['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].max()
    # p = voc_p[voc_p['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].max()
    # 收费号码一天最多呼入几次  降
    # n = voc_n[voc_n['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
    #     'phone_no_m')['nn'].max()
    # p = voc_p[voc_p['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
    #     'phone_no_m')['nn'].max()
    # 收费号码每天平均通话次数  降
    # n = voc_n.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].mean()
    # p = voc_p.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].mean()
    # 收费号码每天平均呼出次数 略降
    # n = voc_n[voc_n['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].mean()
    # p = voc_p[voc_p['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].mean()
    # 收费号码每天平均呼入次数  降
    # n = voc_n[voc_n['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
    #     'phone_no_m')['nn'].mean()
    # p = voc_p[voc_p['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
    #     'phone_no_m')['nn'].mean()
    # 可增加日平均呼入呼出比，以上日平均都是有通话记录的平均值，增加通话记录除以总天数的平均值计算方法

    # 收费号码总通话时长  略降
    # n = voc_n.groupby('phone_no_m')['call_dur'].sum()
    # p = voc_p.groupby('phone_no_m')['call_dur'].sum()
    # 收费号码日均通话时长
    # n = voc_n.groupby(['phone_no_m', 'date'])['call_dur'].mean()
    # p = voc_p.groupby(['phone_no_m', 'date'])['call_dur'].mean()
    # 收费号码日均呼出时长
    # n = voc_n[voc_n['calltype_id'] == 1].groupby(['phone_no_m', 'date'])['call_dur'].mean()
    # p = voc_p[voc_p['calltype_id'] == 1].groupby(['phone_no_m', 'date'])['call_dur'].mean()
    # 收费号码日均呼入时长
    # n = voc_n[voc_n['calltype_id'] == 2].groupby(['phone_no_m', 'date'])['call_dur'].mean()
    # p = voc_p[voc_p['calltype_id'] == 2].groupby(['phone_no_m', 'date'])['call_dur'].mean()

    # 号码拨出五秒钟被拒绝率

    # g = sns.kdeplot(n, color='Red', shade=True)
    # g = sns.kdeplot(p, color='Green', shade=True)
    # g.set_xlabel('收费号码日均呼入时长')
    # g.set_ylabel("Frequency")
    # g = g.legend(["negative", "positive"])
    # plt.show()


def after_handle(test_phone_list, dataset):
    # 后处理
    # test_phone_list: 电话号码列表; dataset: 要分析的数据集
    # 有回拨或回复短信行为的电话认为是非诈骗电话
    # print('Start')
    global g_res_set
    phone_opposite = dataset[['phone_no_m', 'opposite_no_m']]
    res_set = dict()
    res_set_2 = dict()

    for phone in tqdm(test_phone_list):
        opp_phones = phone_opposite[phone_opposite['phone_no_m'] == phone]['opposite_no_m'].drop_duplicates().tolist()
        opp_phones_count = len(phone_opposite[phone_opposite['phone_no_m'] == phone]['opposite_no_m'].tolist())
        if phone in res_set_2:
            res_set_2[phone] = len(opp_phones)/opp_phones_count  # 重复电话占比
        else:
            pass
        for item in opp_phones:
            recall = phone_opposite[(phone_opposite['phone_no_m'] == item) & (phone_opposite['opposite_no_m'] == phone)].shape[0]
            if recall == 0:
                continue
            else:
                # print('回拨数:%s' % recall)
                if phone in res_set:
                    res_set[phone] += recall
                    # g_res_set[phone] += recall
                else:
                    res_set[phone] = recall
                    # g_res_set[phone] = recall
    return res_set, res_set_2


def analysis():
    time = 'test'
    train1 = pd.read_csv('resource/train_test_res/%s/train.csv' % time)

    # 加入回拨率特征
    recall_train = pd.read_csv('resource/analysis_files/train_recall.csv')[['phone_no_m', 'recall', 'call']]  # recall
    # recall_train['re_ratio'].apply(lambda x: tmp_h(x))
    # recall_train['call'].apply(lambda x: tmp_h2(x))
    # recall_train['kkk'] = recall_train['call'] * recall_train['re_ratio']
    # recall_train = recall_train[['phone_no_m', 'kkk']]
    # train1 = pd.merge(recall_train, train1, on='phone_no_m', how='outer')

    train1.fillna(0, inplace=True)
    train1.drop(['county_name', 'city_name'], axis=1, inplace=True)
    test1 = pd.read_csv('resource/train_test_res/%s/test.csv' % time)

    # 加入回拨率特征
    recall_test = pd.read_csv('resource/analysis_files/test_recall.csv')[['phone_no_m', 'recall', 'call']]  # recall
    # recall_test['call'].apply(lambda x: tmp_h2(x))
    # recall_test['kkk'] = recall_test['call'] * recall_test['re_ratio']
    # recall_test = recall_test[['phone_no_m', 'kkk']]
    # test1 = pd.merge(recall_test, test1, on='phone_no_m', how='outer')

    test_id = test1['phone_no_m'].tolist()
    test1.drop(['phone_no_m'], axis=1, inplace=True)
    test1.fillna(0, inplace=True)

    train_col = train1.columns.to_list()
    train_col.remove('label')
    train_col.remove('phone_no_m')
    test1.rename(columns={"arpu_202004": "arpu_"}, inplace=True)
    test1 = test1[train_col]

    c_list = ['calltype_id']  # 'city_name', 'county_name',
    x_train, x_test, y_train, y_test = train_split(train1)


    # 交叉验证
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2020)  # 3折交叉
    purchase_result = 0  # 最终结果
    data_k, label_k = train_split_k(train1)
    for train_index, test_index in kf.split(data_k, label_k):
        X_train_p, y_train_p = data_k.iloc[train_index], label_k.iloc[train_index]
        X_valid_p, y_valid_p = data_k.iloc[test_index], label_k.iloc[test_index]
        purchase_result += cb_model(X_train_p, X_valid_p, y_train_p, y_valid_p, test1, c_list) / 3.0
    print(purchase_result)
    res_dict = {'phone_no_m': test_id, 'label': [value-0.3 for value in purchase_result]}  # round(value-0.3)
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv('./resource/train_test_res/%s/res_cb_3_-0.4_prob.csv' % time, index=False)

    # 开始训练
    # res = cb_model(x_train, x_test, y_train, y_test, test1, c_list)
    # predictions = [round(value) for value in res]
    # res_dict = {'phone_no_m': test_id, 'label': predictions}
    # res_df = pd.DataFrame(res_dict)
    #
    # res_df.to_csv('./resource/train_test_res/%s/res.csv' % time, index=False)


def get_recall(data_df, type, filename):
    in_col = ''
    out_col = ''
    if type == 'call':
        in_col = 'in_dict'
        out_col = 'out_dict'
    elif type == 'sms':
        in_col = 'in_dict_sms'
        out_col = 'out_dict_sms'
    else:
        print('ERROR')
        return
    id2num, num2id = pickle.load(open('resource/xff/dict_2.pkl', 'rb'))
    xff = pickle.load(open('resource/xff/feature_2.pkl', 'rb'))
    phone_list = data_df['phone_no_m'].drop_duplicates().tolist()  # 编码形式的电话号

    # for i in phone_list[:3]:
    #     print(xff.iloc[num2id[i]])
    #
    # return

    # 没有拨出记录的电话号码list
    no_list = []
    print(len(phone_list))
    for i in phone_list:
        if len(xff.iloc[num2id[i]][out_col]) == 0:
            no_list.append(i)
    print('缺失数量%s' % len(no_list))
    print(no_list)

    recal_count = {}
    for phone in tqdm(phone_list):
        opp = xff.iloc[num2id[phone]][out_col].keys()  # 号码phone拨出的所有电话list
        phone_in = xff.iloc[num2id[phone]][in_col]  # 号码phone打入的所有号码dict
        re_count = 0
        for p_ in opp:
            if p_ in phone_in:
                re_count += 1  # xff.iloc[num2id[p_]]['out_dict'][num2id[phone]]
            else:
                continue
        recal_count[phone] = (re_count, len(opp))  # 回拨电话数，拨出电话数


    # 修改没有拨打记录电话号的拨打数和回拨数
    train_phone = load_data('train_voc')[['phone_no_m', 'opposite_no_m']]
    test_phone = load_data('test_voc')[['phone_no_m', 'opposite_no_m']]
    phones = pd.concat([train_phone, test_phone], ignore_index=True)
    mis_dict = {}
    for index, mistake_phone in tqdm(enumerate(no_list)):
        opp_phones_tmp = phones[phones['phone_no_m'] == mistake_phone]['opposite_no_m'].drop_duplicates().tolist()
        call_tmp = len(phones[phones['phone_no_m'] == mistake_phone]['opposite_no_m'].tolist())  # 所有拨出的电话
        tmp = 0
        for item in opp_phones_tmp:
            recall = phones[(phones['phone_no_m'] == item) & (phones['opposite_no_m'] == mistake_phone)].shape[0]  # 回拨电话数
            if recall == 0:
                continue
            else:
                tmp += recall
        mis_dict[mistake_phone] = (tmp, call_tmp)


    # 缺失值字典
    # mis_dict = {'6c64e518c3e8c2163ace9470fc740619e0b68b56a1865679f9aad61e74117bbd18a2eb43f9f20d3d5940d3ebc7718119d99ab43bce7c5b0a0a1e9a938c1f80f3': (0, 215), 'cb0e22e4e913a1e935eed6b4c00d2305da8762e462c0320cc10caeaeed1c4d444e59ac41bf1351de4ddb2b852841dfddd70c6a13bad37843963ac34846f6cdb8': (0, 3), '7622e27f2e18d656f261fbd8942d709199ab1ecbbc0d993b8cd6c20b4b1fad6a6e04856943579bfd89b04c9e22802ca2be02c4e5d7e3c624ccda7fdc82fee31f': (0, 33), 'f820422e6932819e6f847480e31fb78bac8df34e4e0c25b6bf66ee2e23a21f0011d24658156f1f52b5114922abef7b65b00fc067dd603e17a4fdbb86345d7f1c': (0, 14), 'bdeaae7d578b47385cded8f585ef50933d2ef7e00e2dbb1b770fe2a386b5e28d622319fe59ac265d6bdcc1aee5929120785166cda7d0e91e5f2cd09bf3512a53': (0, 6), '86f6dda90d57c89ec65d63e9be82ecaa9532492f824356b7e7139bca31cd1feb61466b12a1c99b352e3235dfe82e78dfab74cb73430cbc68e734256d8b36715f': (0, 40), 'd2121a4720f94cbcf5afb8813baf4f1cd917456f61712eb92c8654314417cc01ab4a7311a58d42326032401eca96f6bc3d6fba4b546130f11bd3cdbc4ae09508': (0, 4), 'e2b33981da8323e4353265d36b21244f763429f4b3f2a66ddb103bc45328af968dc0ffddfd59293f042740183f5e5452da5d64a6ff9b4a21ea8f4135d2031ba1': (0, 174), '62fe677b773a12397c14a6bf22c31ec97e6bc68f55bba59ff8ba263694e39b0e8a2ab9a6a60fd7ef9779a655e8ef30f1a6e8997812f987e51c4f20f5cb9fb891': (0, 1), '6a82a5343744351425214fe5f43eb37257e15ec2f5c0accc969508fa2686cae006900248327c8c9c028d512c9f0a06c05a7b5ebff5ad54e6ea5c31699f8cf95b': (0, 4), 'ee8fe686bf8b59972d4f2da1072ccc73fd58b9e21c2054bc5699aa6589adbe237ce40b6a243a38600ca9e2ed26463bd20d67e3c367b59da0f2b4cdefebd805ef': (0, 3), '4e25e61334b873cb393a35c8a574b3b516e93b43522c1874d9fbf01d0de268f3c55eb4482cdb1fa33bd66e882cecb0a11dc5b80f131aeb8b2987c0ae7f7d64ff': (0, 1), 'c29e3bd9d23fffd9e56783a0fc2214ea398022d2021f3be451f352e73b9386af1ccf0651509af0be72cc982ae98b291f5bd06fa916ac47b7cb921256bad0d966': (0, 4), '06c07c3f66743e506b706a358be3478e2c49aa038e479576c5f8c7b839f6b9ad4f1d52b0f85dcdd7f55579c773cf764388848ce5d7c4bd2d80dcbcf044fb2fb1': (0, 3), 'd2628e3d21a0154bf9305ed8dde581592dcac611c0fa484927802eedb138b23583a46c6c42ff7658309c9387fc1e8423866a9c9e32753cde8a4c4dd956d58d0f': (0, 1), '05cc0954aa9ac2269e6b56365d29bf8d7ad2c260061352ec468beb7707d98f1dd7fab0c68e6692bca9488b111953174a5da7a5fe92fc373cc5b032ca1520761a': (0, 62), 'f519b5de977ccbe3b541d6ee750f6dbb0c7fd9c02770833f804115e110f7a6f500c1b69d1ebe676b17dd19cc057f82ec5a024d18ab4a72ec5e11f890a95c9840': (0, 95), 'd3052f74b7e510c55dee261c8da844948c193de160263e7dff8e5c5a26db418a3c3fc8a3245e0ccf822b3e0c4b0cce247154cd75c9f7376ee104b675f0680f2a': (0, 53), '5e2dfd3432984f0b24376ec5e208fc8517928c69887dd16ae4f31a3146c773d109a5bde1c9c7a8b883b1e33ca8ad24a056e9cc860f2377e21ab627f7eec8953e': (0, 102), '950d19e8d7c6a6f3bc912b15b85941db64af1786b3e6657caaf9fac228d3bfe87b466371892d50e5fd679cbdc45d919a8ce6cd7c2e39881a41304fba49e49197': (0, 22), 'f33af4116ff96e9410d7979ada3e484a4d9648d8f98f3a3640bf58981465b177d1b44a57d2a18b24a915dc0620cd739bf42a1f91d259137e20cf0270a833e1de': (0, 45), 'f751188e472d8f343d4d816f762fe5bbecd04883f75564f850dc56eab78692ded6edf24a6f2a18a88a7ed9eec8acccd645749ea59756583ae2fe32591a55005b': (0, 15), 'fa6dd823e151715b660f4f36aa722cee6c12a7468bc615677ca202da0e2d1669b80fdbef181164835268d34e73fe6281e7f72a57ea8dc38d2c5e5c81333ab557': (0, 1), '9fc0d282aa1365d32a8e4d2c4d41dd0d389e8447428834e0e4d789b73219e9626e8aee76c63950d4292eb56b33a71d2f9dda45338c6e8a7e7f2d08191d32b871': (0, 177), '67a0a9ca48d440f85497b82d40488503658ec60c680b364d0e2ee1f88458448e53c9e923d4e60b2181d1b09b728d26eed4a0824d9e5c8090fcbeeedff21f8c46': (0, 156), 'f3300703dab4cd7507106f32c700fe52c12c6bc95854cba0ea01468ceb3a2806aed7849de0d29d1f3e0f069bbeaad64c175db99b1e2cdf4847d2b38bf01b88ad': (0, 60), 'c4e9eeb1d3ac8d1daea27664b3cccd1cad02def3b29585f4855efa03ee255b28a918980b6076a58bf95c915803cd382a07a4d3133392a9c577d3dd7f3345986b': (0, 22), '1879f3ebb34b070b97885953410b0681d814dcb60ea17b35be08c6691b6d247654f389d8fe3c13d09099937c73e7572d30e96db5e437c34357e858e76a0319ed': (0, 17), '74b1533fd7596654ecf8be09db8b7ad7fc1e89b8424424d125cd9d8c1c5e733c163ac36ed9a24ee1864baba01440ca30eb72eae422f97279940c24afd1b7be7f': (0, 1), '9e12cec51eb5b7f51a973adabcf0e044b9560fbbe70756eb4936af3cdd8ef4cf5b63573e2831e0713d7896b88b0da19d9299ac33e6a5db3b8bacb393d0235e24': (0, 277), '8bef795f6bd951a59136e4b7d8a28bccd9c38d8724436d1c240fc6a00b4cf0029a8e6b10a5d8b9ba5a1565d6fff6e7a84a0397fa08c2ece9e0de76fbbe4e2d99': (0, 78), 'dea4220801f66ec39ad0ed6de76d429a1b566f19bf72a77b6d0aefb57b9b44ebb1673daa2695817f3d7d282ab9ff8a835cf8dff31a7aed5c60f36a3705845c82': (0, 3), '70fb47148e0d0cc772ee72482dc265762c5be0facc8c7a1c20d0191a808731c89774966c6153a3727fb996183db3385bcc01e69f73e39a4595fb42f7309e09ac': (0, 2), '4a66f8a8488c58c618542b5458b6535aafaa9a191ac06e36397dc7d9a3dcd37442ea2d54ae534359cafee07f783255f92728950a339e88ac8369c80df8b789ed': (0, 1), '233858642690d8a6153762d8ebce0b2bcc25c867ec6f988e09f625b2df0c603d22b1a6ff4fc7600cd31eaba172dbf556f734fd851a8fcbc6c93f52ce4ab7d6e5': (0, 2), '9e1b59525baeb052df87d34c5ee0cc7e6be7c717f072a71b74c71e6f1a2960a88a46f73fa0ce81447c9bfd6d58665cd4c929562b2c5ec7e80c51e3e75c129509': (0, 1)}
    for k, v in mis_dict.items():
        recal_count[k] = v

    re = [x[0] for x in list(recal_count.values())]
    call = [x[1] for x in list(recal_count.values())]
    re_ratio = [x[0]/x[1] for x in list(recal_count.values())]
    save_dict = {'phone_no_m': list(recal_count.keys()), 'recall': re, 'call': call, 're_ratio': re_ratio}
    save_df = pd.DataFrame(save_dict)
    save_df.to_csv('resource/analysis_files/%s.csv' % filename, index=False)


def recall(phone_list):
    recal_count = {}
    xff = pickle.load(open(xff_path + 'new_feature.pkl', 'rb'))
    for phone in phone_list:
        opp = xff.iloc[phone]['out_dict'].keys()  # 号码phone拨出的所有电话list
        phone_in = xff.iloc[phone]['in_dict']  # 号码phone打入的所有号码dict
        re_count = 0
        for p_ in opp:
            if p_ in phone_in:
                re_count += 1  # xff.iloc[num2id[p_]]['out_dict'][num2id[phone]]
            else:
                continue
        recal_count[phone] = (re_count, len(opp))  # 回拨电话数，拨出电话数
    return recal_count


def triple_friends(phone_list):
    t_count = {}
    # for phone in phone_list:


def after_week_recall(voc, t):
    # 时间列转化
    voc['start_datetime'] = pd.to_datetime(voc['start_datetime'])
    voc["year"] = voc['start_datetime'].dt.year
    voc["month"] = voc['start_datetime'].dt.month
    voc["day"] = voc['start_datetime'].dt.day
    voc['start_ymd'] = pd.to_datetime(voc[['year', 'month', 'day']])
    # 电话转换为id
    phone_list = [num2id[x] for x in voc['phone_no_m'].drop_duplicates().tolist()]
    # 回拨电话字典
    recal_count = {}  # {'拨出id': [回拨id1, 回拨id2, 回拨id3 ...]}
    for phone in phone_list:
        opp = xff.iloc[phone]['out_dict'].keys()  # 号码phone拨出的所有电话list
        phone_in = xff.iloc[phone]['in_dict']  # 号码phone打入的所有号码dict
        recall_phones = set()
        for p_ in opp:
            if p_ in phone_in:
                recall_phones.add(p_)
            else:
                continue
        recal_count[phone] = list(recall_phones)  # 打回来的电话list

    res = {}
    outlayer_id = set()  # 字典缺失值
    for i in tqdm(recal_count.keys()):
        count = 0
        if len(recal_count[i]) == 0:
            continue
        for j in recal_count[i]:
            i, j = int(i), int(j)
            try:
                # i第一次给j通话的日期
                earliest = voc[(voc['phone_no_m'] == id2num[i]) & (voc['opposite_no_m'] == id2num[j])][
                    'start_ymd'].min()
                #
                # after_week = voc[(voc['phone_no_m'] == id2num[i]) & (voc['opposite_no_m'] == id2num[j]) &
                #                     (voc['start_ymd'] > (earliest + pd.Timedelta(days=7)))]['opposite_no_m'].drop_duplicates().shape[0]
                # 7天后有没有拨回来
                after_week_recall = voc[(voc['phone_no_m'] == id2num[j]) & (voc['opposite_no_m'] == id2num[i]) &
                                           (voc['start_ymd'] > (earliest + pd.Timedelta(days=7)))]['opposite_no_m'].drop_duplicates().shape[0]
                count += after_week_recall
            except KeyError as e:
                outlayer_id.add(e)
        res[id2num[i]] = count
    print('缺失数量', len(outlayer_id))
    print([x for x in res.values() if x != 0])
    pickle.dump(res, open(xff_path + 'after_week_recall_%s.pkl' % t, 'wb'))


def after_week_call(voc, t):
    voc['start_datetime'] = pd.to_datetime(voc['start_datetime'])
    voc["year"] = voc['start_datetime'].dt.year
    voc["month"] = voc['start_datetime'].dt.month
    voc["day"] = voc['start_datetime'].dt.day
    voc['start_ymd'] = pd.to_datetime(voc[['year', 'month', 'day']])
    #
    phone_list = voc['phone_no_m'].drop_duplicates().tolist()
    res = {}
    for i in tqdm(phone_list):
        opp = voc[voc['phone_no_m'] == i]['opposite_no_m'].drop_duplicates().tolist()
        cnt = 0
        for j in opp:
            first = voc[(voc['phone_no_m'] == i) & (voc['opposite_no_m'] == j)]['start_ymd'].min()
            after_week_call_count = voc[(voc['phone_no_m'] == i) & (voc['opposite_no_m'] == j) &
                                        (voc['start_ymd'] >= (first + pd.Timedelta(days=7)))].shape[0]
            cnt += after_week_call_count
        res[i] = cnt
    print([x for x in res.values() if x != 0])
    pickle.dump(res, open(xff_path + 'after_week_call_%s.pkl' % t, 'wb'))

if __name__ == '__main__':
    # handel_dataset('test', 'train')
    # handel_dataset('test', 'test')
    # analysis()

    id2num, num2id = pickle.load(open(xff_path + 'new_dict.pkl', 'rb'))
    xff = pickle.load(open(xff_path + 'new_feature.pkl', 'rb'))

    train_voc = load_data('train_voc')
    test_voc = load_data('test_voc')
    p_voc = pd.read_csv(analysis_path + 'voc_positive.csv')
    n_voc = pd.read_csv(analysis_path + 'voc_negative.csv')

    # after_week_recall(n_voc, 'n')
    # after_week_recall(train_voc, 'train')
    # after_week_recall(test_voc, 'test')
    # after_week_recall(p_voc, 'p')

    # after_week_call(n_voc, 'n')
    # after_week_call(train_voc, 'train')
    # after_week_call(test_voc, 'test')
    # after_week_call(p_voc, 'p')

    threads = []
    threads.append(threading.Thread(target=after_week_call(n_voc, 'n')))
    threads.append(threading.Thread(target=after_week_call(train_voc, 'train')))
    threads.append(threading.Thread(target=after_week_call(test_voc, 'test')))
    threads.append(threading.Thread(target=after_week_call(p_voc, 'p')))
    threads.append(threading.Thread(target=after_week_recall(n_voc, 'n')))
    threads.append(threading.Thread(target=after_week_recall(train_voc, 'train')))
    threads.append(threading.Thread(target=after_week_recall(test_voc, 'test')))
    threads.append(threading.Thread(target=after_week_recall(p_voc, 'p')))

    for t in threads:
        t.start()

