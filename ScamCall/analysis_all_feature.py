import pandas as pd
from ScamCall.analysis_best import xff_path, analysis_path, load_data


'''所有特征'''


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

    # data_set.to_csv(analysis_path + '%s_%s.csv' % (time, c), index=False)
    # print('%s_%s.csv写入完成' % (time, c))

    return data_set


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
    busi_name_nunique = df.groupby('phone_no_m', sort=False)['busi_name'].nunique().reset_index(
        name='busi_name_nunique')
    flow_sum = df.groupby('phone_no_m', sort=False)['flow'].sum().reset_index(name='flow_sum')

    # 可以尝试增加每月流量变化幅度的特征
    app_month = df.groupby('phone_no_m')['month_id'].nunique().reset_index(name='app_month')

    new_df1 = pd.merge(busi_name_nunique, flow_sum, on='phone_no_m', how='outer')
    new_df = pd.merge(new_df1, app_month, on='phone_no_m', how='outer')


    # new_df.to_csv('./resource/app_%s_1.csv' % c, index=False)
    return new_df


def handel_voc(df):
    # 通话信息处理

    # 拆分时间列为日期和时间
    df['date'] = df['start_datetime'].map(lambda x: x.split(' ')[0])
    df['time'] = df['start_datetime'].map(lambda x: x.split(' ')[1])
    # df['date'] = df['start_datetime'].map(lambda x: x.split(' ')[0])
    # df['time'] = df['start_datetime'].map(lambda x: x.split(' ')[1])

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


    # 一个手机号的通话记录数
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
    averge_call = df.groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby('phone_no_m')['nn'].mean().reset_index(name='averge_call')
    # 号码平均每天呼出次数  略降
    averge_call_01 = df[df['calltype_id'] == 1].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
        'phone_no_m')['nn'].mean().reset_index(name='averge_call_01')
    # 号码平均呼入次数  降
    averge_call_02 = df[df['calltype_id'] == 2].groupby('phone_no_m')['date'].value_counts().reset_index(name='nn').groupby(
        'phone_no_m')['nn'].mean().reset_index(name='averge_call_02')
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


    # 一个手机用过的电话卡数量
    # imei_phones = df.groupby('imei_m')['phone_no_m'].nunique().reset_index(name='imei_phones_count')
    # tmp_dic = dict(zip(imei_phones['imei_m'].tolist(), imei_phones['imei_phones_count'].tolist()))
    # imei_phones_c = df.groupby('phone_no_m')['imei_m'].value_counts().reset_index(name='tmp')


    m_list = [oppsite_no_m_voc_nunique, call_diff_01, call_diff_02, start_datetime_count, date_unique,
              call_dur_mean, city_name_nunique, county_name_nunique, imei_m_nunique]
    m_list_other = [tmp_ratio, dure_sum, call_sum, call_sum_01, call_sum_02, date_unique_01, date_unique_02,
                    call_day_max, call_day_01_max, call_day_02_max, averge_call, averge_call_01, dure_std,
                    dure_mean, phone_count_max, imei_county_mean, averge_call_02]

    total = m_list+m_list_other
    new_df = pd.merge(total[0], total[1], on='phone_no_m', how='outer')
    for i in range(2, len(total)):
        new_df = pd.merge(new_df, total[i], on='phone_no_m', how='outer')

    # new_df.to_csv('./resource/voc_%s_1.csv' % c, index=False)
    return new_df


def main():
    return handel_dataset('all', 'train'), handel_dataset('all', 'test')


