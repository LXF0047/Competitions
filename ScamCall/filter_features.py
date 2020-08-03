from ScamCall.analysis_best import load_data, recall
import pandas as pd
import pickle


def rule(t):
    '''
    流量使用月数为0或1 1689
    拨出+接听不同电话大于100 2513
    拨出100个电话以上 1811
    回拨率不为0 4518
    手机换卡次数大于10 118
    '''
    if t == 'test':
        df_app = load_data('test_app')
        df_voc = load_data('test_voc')
    elif t == 'train':
        df_app = load_data('train_app')
        df_voc = load_data('train_voc')
    else:
        print('输入test或train')
        return None
    # 加载字典数据
    xff_path = '/home/lxf/data/xff/'
    id2num, num2id = pickle.load(open(xff_path + 'new_dict.pkl', 'rb'))

    # 流量使用月数为0或1
    month_p = df_app.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')
    flow_01 = month_p[month_p['month_count'] < 2]['phone_no_m'].to_list()
    print('流量使用月数为0或1', len(flow_01))

    # 拨出+接听不同电话大于100
    # call_100 = df[df['phone_no_m'].isin(p_phone)].groupby('phone_no_m')[
    #     'opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_take_100 = df_voc.groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_take_100_phone = call_take_100[call_take_100['voc_nunique'] > 100]['phone_no_m'].to_list()
    print('拨出+接听不同电话大于100', len(call_take_100_phone))

    # 拨出100个电话以上
    call_100 = df_voc[df_voc['calltype_id'] == 1].groupby('phone_no_m')['opposite_no_m'].nunique().reset_index(name='voc_nunique')
    call_100_phone = call_100[call_100['voc_nunique'] > 100]['phone_no_m'].to_list()
    print('拨出100个电话以上', len(call_100_phone))

    # 回拨率
    phone_id_list = [num2id[x] for x in df_voc['phone_no_m'].drop_duplicates().tolist()]
    recall_count = recall(phone_id_list)  # {'phone': (回拨数, 拨出数)}
    total = sum([recall_count[x][1] for x in recall_count])  # 所有拨打数，用来惩罚拨出少回拨多的
    n_ratio = [(recall_count[x][0] / recall_count[x][1]) * (recall_count[x][1] / total) for x in recall_count.keys() if
               recall_count[x][1] != 0]
    n_ra_0 = len([x for x in n_ratio if x != 0])
    print('回拨率不为0', n_ra_0)

    # 手机换卡次数大于10
    phone_imei_count = df_voc.groupby('phone_no_m')['imei_m'].nunique().reset_index(name='imei_phone')
    phone_imei_count_10 = phone_imei_count[phone_imei_count['imei_phone'] > 10]
    print('手机换卡次数大于10', phone_imei_count_10.shape[0])



if __name__ == '__main__':
    train_voc = load_data('train_voc')
    train_user = load_data('train_user')
    # train_sms = load_data('train_sms')
    # train_app = load_data('train_app')
    #
    # voc_phones = train_voc['phone_no_m'].drop_duplicates().tolist()
    # sms_phones = train_sms['phone_no_m'].drop_duplicates().tolist()
    # app_phones = train_app['phone_no_m'].drop_duplicates().tolist()
    # all_phones = train_user['phone_no_m'].drop_duplicates().tolist()
    #
    # notin_voc = list(set(all_phones).difference(set(voc_phones)))
    # notin_sms = list(set(all_phones).difference(set(sms_phones)))
    # notin_app = list(set(all_phones).difference(set(app_phones)))
    #
    # print('voc缺失电话数: %s, 诈骗电话数量: %s' % (len(notin_voc), train_user[train_user['phone_no_m'].isin(notin_voc)]['label'].sum()))
    # print('sms缺失电话数: %s, 诈骗电话数量: %s' % (len(notin_sms), train_user[train_user['phone_no_m'].isin(notin_sms)]['label'].sum()))
    # print('app缺失电话数: %s, 诈骗电话数量: %s' % (len(notin_app), train_user[train_user['phone_no_m'].isin(notin_app)]['label'].sum()))
    #
    # # test_voc = load_data('test_voc')
    # # test_all = load_data('test_user')
    # #
    # # test_voc_phones = test_voc['phone_no_m'].drop_duplicates().tolist()
    # # test_phones_all = test_all['phone_no_m'].drop_duplicates().tolist()
    # # test_notin_voc = list(set(test_phones_all).difference(set(test_voc_phones)))
    # #
    # # print('测试集中缺失值数量%s' % len(test_notin_voc))
    #
    # print(train_user[train_user['phone_no_m'].isin(notin_voc)]['label'])

    rule('train')