from ScamCall.filter_features import bool_feature, num_feature, stack
from ScamCall.baseline089 import feats
from ScamCall.analysis_all_feature import main


def generate_all_feature():
    train1, test1 = main()
    train2, test2 = feats()
    train3, test3 = bool_feature('train'), bool_feature('test')
    train4, test4 = num_feature('train'), num_feature('test')

    print(train1.shape[0], train2.shape[0], train3.shape[0], train4.shape[0])