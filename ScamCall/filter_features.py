from ScamCall.analysis_best import load_data
import pandas as pd


def rule(df):
    month_p = df.groupby('phone_no_m')['month_id'].nunique().reset_index(name='month_count')

    
if __name__ == '__main__':
    pass