import pandas as pd
from ScamCall.analysis_best import analysis_path, cb_model, train_split

train = pd.read_csv(analysis_path + 'all_features_train.csv')
test = pd.read_csv(analysis_path + 'all_features_test.csv')
test.rename(columns={"arpu_202005": "arpu_"}, inplace=True)
train.drop(['city_name', 'county_name', 'label_y', 'label_x'], axis=1, inplace=True)
test.drop(['city_name', 'county_name', 'phone_no_m'], axis=1, inplace=True)

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
test.loc[test['arpu_'] == '\\N', 'arpu_'] = 0

x_train, x_test, y_train, y_test = train_split(train)

res = cb_model(x_train, x_test, y_train, y_test, test, [])
pre = [round(x) for x in res]
print(sum(pre))