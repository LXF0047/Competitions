from ScamCall.analysis_best import load_data, recall, analysis_path, xff_path
import pandas as pd
import pickle
import prettytable as pt


def rule(t):
    '''
    总共 6025
    流量使用月数为0或1 1689
    拨出+接听不同电话大于100 2513
    拨出100个电话以上 1811
    回拨率不为0 4518
    手机换卡次数大于10 118
    缺失值
    '''
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
        return None
    # 加载字典数据
    id2num, num2id = pickle.load(open(xff_path + 'new_dict.pkl', 'rb'))

    # 表格打印
    tb = pt.PrettyTable()
    tb.field_names = ["说明", "Count", "Total"]

    # 流量使用月数为0或1
    month_p = df_app.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')
    flow_01 = month_p[month_p['month_count'] < 2]['phone_no_m'].to_list()
    tb.add_row(["流量使用月数为0或1", len(flow_01), df_app['phone_no_m'].drop_duplicates().shape[0]])
    # print('流量使用月数为0或1', len(flow_01))

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

    print(tb)


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


if __name__ == '__main__':
    '''
    +---------------------------------------+-------+-------+
    |                  说明                 | Count | Total |
    +---------------------------------------+-------+-------+
    |           流量使用月数为0或1          |  1196 |  1962 |
    |        拨出+接听不同电话大于150       |  888  |  1892 |
    |           拨出200个电话以上           |  630  |  1876 |
    |         主叫100次以上回拨率为0        |  498  |  1038 |
    |           手机换卡次数大于15          |   52  |  1892 |
    |            主叫通话时长为0            |   7   |   7   |
    | 固定通话个数（打过10次以上的）大于100 |   4   |  1892 |
    |   相同号码拨出电话的最多次数大于100   |  181  |  1892 |
    +---------------------------------------+-------+-------+
    
    +---------------------------------------+-------+-------+
    |                  说明                 | Count | Total |
    +---------------------------------------+-------+-------+
    |           流量使用月数为0或1          |  493  |  4144 |
    |        拨出+接听不同电话大于150       |  919  |  4133 |
    |           拨出200个电话以上           |  244  |  4113 |
    |         主叫100次以上回拨率为0        |   51  |  774  |
    |           手机换卡次数大于15          |   2   |  4133 |
    |            主叫通话时长为0            |   0   |   0   |
    | 固定通话个数（打过10次以上的）大于100 |   4   |  4133 |
    |   相同号码拨出电话的最多次数大于100   |  1272 |  4133 |
    +---------------------------------------+-------+-------+
    '''
    rule('n')
    print('*' * 100)
    rule('p')
