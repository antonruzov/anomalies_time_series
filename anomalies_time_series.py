import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')


def double_exponential_smoothing(series, alpha, beta):
    """
    Функция двойного экспоненциального сглаживания.
    """
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # прогнозируем
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result[:-1]


def fur(y, k=8):
    """
    Функция сглаживания периодической функцией.
    """
    y = y.copy()
    n = len(y)
    y = y.reshape(-1, 1)
    x = np.arange(1, n+1).reshape(-1, 1)
    
    x_ones = np.hstack((np.ones(x.shape), x))
    A = np.dot(np.linalg.inv(np.dot(x_ones.T, x_ones)), np.dot(x_ones.T, y))
    a, b = A[1:], A[0][0]
    linear = a * x + b
    
    y_minus_linear = y - linear
    cos = np.hstack(([np.cos(2*np.pi*k * x/n) for k in range(1, k+1)]))
    sin = np.hstack(([np.sin(2*np.pi*k * x/n) for k in range(1, k+1)]))
    a_k = np.sum(y_minus_linear * cos, axis=0) * 2 / n
    b_k = np.sum(y_minus_linear * sin, axis=0) * 2 / n
    s = (np.sum(a_k * cos, axis=1) + np.sum(b_k * sin, axis=1)).reshape(-1, 1) + linear
    
    return s
    

if __name__ == '__main__':

    df = pd.read_csv('SBER_190101_200101_15_minute.csv',
                                         sep = ';',
                                         index_col=0,
                                         parse_dates={'Date&Time': [0, 1]}).sort_index()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek + 1
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['date'] = df.index.date
    df['date'] = df['date'].astype(str)
    
    df_aggregate = pd.DataFrame(df.groupby(['date', 'hour'])['Volume'].agg([np.average]))
    df_test = df[df['date'] == '2019-12-30']
    #df_test['rolling_mean_volume'] = df_test['Volume'].rolling(window=3).mean()
    
    #Сглаживаем тестовую выборку.
    INTERVAL = 0.5
    volume = df_test['Volume'].values.reshape(-1, 1)
    fourier = fur(volume, k=4)
    df_test['fur'] = fourier
    upper_bond = fourier + fourier * INTERVAL
    lower_bond = fourier - fourier * INTERVAL
    anomalies = np.array([np.NaN]*len(volume)).reshape(-1, 1)
    anomalies[volume<lower_bond] = volume[volume<lower_bond]
            
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title('Fourier smoothing')
    ax.plot(df_test.index, volume, label='volume')
    ax.plot(df_test.index, fourier, label='Fourier')
    ax.fill_between(df_test.index, upper_bond.reshape(-1, ), lower_bond.reshape(-1, ), alpha=0.5, color='grey')
    ax.plot(df_test.index, anomalies, markersize=10, color='red', marker='o')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    
    df_test['Anomalies'] = anomalies
    for index, row in df_test[df_test['Anomalies'].notnull()].iterrows():
        dayofweek = row['dayofweek']
        hour = row['hour']
        minute = row['minute']
        value = row['Volume']
        values_current_time = df[(df['dayofweek']==dayofweek) & (df['hour']==hour) & (df['minute']==minute)]['Volume']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Day {}, hour {}, minute {}, VALUE_ANOMILIES = {}'.format(dayofweek, hour, minute, value))
        sns.boxplot(x=values_current_time, ax=ax1)
        sns.swarmplot(x=[value], color='red', marker='o', ax=ax1)
        sns.distplot(values_current_time.values, hist=False, ax=ax2)
        plt.xlim([values_current_time.min(), values_current_time.max()])
        plt.show()
        
        
    """
    Выше реализован первый вариант - тот вариант, который обсуждали на совещании.
    Мне кажется, это не то что нужно.
    """
    df_train = df[df['date'] != '2019-12-30']
    df_train_agg = pd.DataFrame(df_train.groupby(['dayofweek', 'hour'])['Volume'].agg([np.sum]))
    df_train_agg['dayofweek'] = [ixd[0] for ixd in df_train_agg.index]
    df_train_agg['hour'] = [ixd[1] for ixd in df_train_agg.index]
    df_train_agg = df_train_agg.reset_index(drop=True)
    df_train = df_train.merge(df_train_agg, left_on=['dayofweek', 'hour'], right_on=['dayofweek', 'hour'])
    df_train['norm'] = df_train['Volume'] / df_train['sum']
    
    df_test = df_test.merge(df_train_agg, left_on=['dayofweek', 'hour'], right_on=['dayofweek', 'hour'])
    df_test['norm'] = df_test['Volume'] / df_test['sum']
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title('Distribution normalize value')
    ax = sns.distplot(df_train['norm'], hist=False)
    plt.vlines(df_test['norm'], 0, 150, color='red', linestyles='dotted')
    plt.xlim([0, 0.02])
    plt.xlabel('normalize value')
    plt.grid(True)
    plt.show()
    
    
    """
    Плотность потока.
    """
    df_train['cumsum'] = None
    for date in df_train['date'].unique():
        df_train.loc[df_train['date'] == date, 'cumsum'] = df_train[df_train['date'] == date]['Volume'].cumsum().values
    df_test['cumsum'] = None
    for date in df_test['date'].unique():
        df_test.loc[df_test['date'] == date, 'cumsum'] = df_test[df_test['date'] == date]['Volume'].cumsum().values
    n_lst = []
    dayofweek_lst = []
    hour_lst = []
    minute_lst = []
    average_volume = []
    averate_volume_test = []
    for i in range(df_test.shape[0]):
        dayofweek = df_test.iloc[i]['dayofweek']
        hour = df_test.iloc[i]['hour']
        minute = df_test.iloc[i]['minute']
        if dayofweek not in dayofweek_lst:
            dayofweek_lst.append(dayofweek)
        if hour not in hour_lst:
            hour_lst.append(hour)
        if minute not in minute_lst:
            minute_lst.append(minute)
        average = df_train[(df_train['dayofweek'].isin(dayofweek_lst)) &
                           (df_train['hour'].isin(hour_lst)) &
                           (df_train['minute'].isin(minute_lst))]['cumsum'].mean()
        average_volume.append(average)
        n_lst.append((i+1) * 15)
        
        average = df_test[(df_test['dayofweek'].isin(dayofweek_lst)) &
                          (df_test['hour'].isin(hour_lst)) &
                          (df_test['minute'].isin(minute_lst))]['cumsum'].mean()
        averate_volume_test.append(average)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(n_lst, average_volume, label='history')
    ax.plot(n_lst, averate_volume_test, label='current')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
        

    
        
        
        
        
        
        
    
    
        
        



 
