from ScamCall.analysis_best import load_data
import pandas as pd


def rule(df):
    month_p = df.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')


if __name__ == '__main__':
    train_voc = load_data('train_voc')
    train_user = load_data('train_user')
    train_sms = load_data('train_sms')
    train_app = load_data('train_app')

    voc_phones = train_voc['phone_no_m'].drop_duplicates().tolist()
    sms_phones = train_sms['phone_no_m'].drop_duplicates().tolist()
    app_phones = train_app['phone_no_m'].drop_duplicates().tolist()
    all_phones = train_user['phone_no_m'].drop_duplicates().tolist()

    notin_voc = list(set(all_phones).difference(set(voc_phones)))
    notin_sms = list(set(all_phones).difference(set(sms_phones)))
    notin_app = list(set(all_phones).difference(set(app_phones)))

    print('voc缺失电话数: %s, 诈骗电话数量: %s' % (len(notin_voc), train_user[train_user['phone_no_m'].isin(notin_voc)]['label'].sum()))
    print('sms缺失电话数: %s, 诈骗电话数量: %s' % (len(notin_sms), train_user[train_user['phone_no_m'].isin(notin_sms)]['label'].sum()))
    print('app缺失电话数: %s, 诈骗电话数量: %s' % (len(notin_app), train_user[train_user['phone_no_m'].isin(notin_app)]['label'].sum()))

    test_voc = load_data('test_voc')
    test_all = load_data('test_user')

    test_voc_phones = test_voc['phone_no_m'].drop_duplicates().tolist()
    test_phones_all = test_all['phone_no_m'].drop_duplicates().tolist()
    test_notin_voc = list(set(test_phones_all).difference(set(test_voc_phones)))

    print('测试集中缺失值数量%s' % len(test_notin_voc))
