import pandas as pd
from math import sin
from math import pi


def prep_data(df):
    tmp = df[['WEEK', 'ACTUAL']].groupby('WEEK', as_index=False)[['ACTUAL']].sum()
    tmp.columns = ['WEEK', 'Revenue']
    tmp['WEEK'] = tmp['WEEK'].apply(lambda x: int(x))
    df['WEEK'] = df['WEEK'].apply(lambda x: int(x))
    df = pd.merge(df, tmp, on='WEEK')
    tmp = df[['WEEK', 'PROMO']].groupby('WEEK', as_index=False)[['PROMO']].sum()
    tmp['Promo'] = tmp['PROMO'].apply(lambda x: 1 if isinstance(x, str) and 'Y' in x else 0)
    tmp = tmp.drop('PROMO', 1)
    df = pd.merge(df, tmp, on='WEEK')
    df['Year'] = df['WEEK'].apply(lambda x: int(str(x)[1:3]))
    df['Date'] = df['WEEK'].apply(lambda x: int(str(x)[3:5] + str(x)[5:7]))
    df = df.drop('WEEK', 1)
    df = df.drop('SEG', 1)
    df = df.drop('ACTUAL', 1)
    df = df.drop('PROMO', 1)
    df = df.drop('BU', 1)
    df = df.drop_duplicates()
    df = df.sort_values(['Year', "Date"])
    df['Month'] = df['Date'].apply(lambda x: x // 300 if x // 300 != 0 else 4)
    df['Season'] = df['Month'].apply(lambda x: sin(2*pi*x/4))
    df = df.drop('Month', 1)
    return df
