# ライブラリ
## 基本
import numpy as np
import matplotlib.pyplot as plt
## 分解
import scipy.stats
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split

# 分解
def bunkai(df, window_size=5):
    ## 不要そうなら消してOK
    # 正規化
    raw_array = scipy.stats.zscore(df)

    ## STL分解
    stl=STL(raw_array, period=12, robust=True)
    stl_series = stl.fit()

    ## STL分解結果のグラフ化
    plt.rcParams['figure.figsize'] = [12, 9]
    stl_series.plot()
    plt.show()

    ## データの整形
    ### default : SlideingWindow = 5
    raw_x, raw_y = sliding_window(raw_array, window_size=window_size)
    trend_x, trend_y = sliding_window(stl_series.trend, window_size=window_size)
    seasonal_x, seasonal_y = sliding_window(stl_series.seasonal, window_size=window_size)
    resid_x, resid_y = sliding_window(stl_series.resid, window_size=window_size)

    ## データの分割
    raw_x_train, raw_x_test, raw_y_train, raw_y_test = train_test_split(raw_x, raw_y, test_size=0.1, shuffle=False)
    trend_x_train, trend_x_test, trend_y_train, trend_y_test = train_test_split(trend_x, trend_y, test_size=0.1, shuffle=False)
    seasonal_x_train, seasonal_x_test, seasonal_y_train, seasonal_y_test = train_test_split(seasonal_x, seasonal_y, test_size=0.1, shuffle=False)
    resid_x_train, resid_x_test, resid_y_train, resid_y_test = train_test_split(resid_x, resid_y, test_size=0.1, shuffle=False)
    
    return [raw_x_train, raw_x_test, raw_y_train, raw_y_test], [trend_x_train, trend_x_test, trend_y_train, trend_y_test], [seasonal_x_train, seasonal_x_test, seasonal_y_train, seasonal_y_test], [resid_x_train, resid_x_test, resid_y_train, resid_y_test]

def sliding_window(target_raw, window_size=5):
    """
    スライディングウィンドウ。

    """
    
    X, Y = [], []
    for i in range(len(target_raw)-window_size):
        X.append(target_raw[i:i + window_size])
        Y.append(target_raw[i + window_size])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# 評価
def mean_squared_error(y_true, y_pred):
    """
    MSEを計算する

    """
    return np.mean((y_true - y_pred) ** 2)

def calc_kld(generated_data, ground_truth, bins=20):
    """
    KLDを計算する

    """
    # 最小値の設定
    generated_data = np.array(generated_data)
    ground_truth = np.array(ground_truth)
    if ground_truth.min() < generated_data.min():
        range_min = ground_truth.min()
    else:
        range_min = generated_data.min()
    
    # 最大値の設定
    if ground_truth.max() < generated_data.max():
        range_max = generated_data.max()
    else:
        range_max = ground_truth.max()
    
    pd_gt, _ = np.histogram(ground_truth, bins=bins, density=True, range=(range_min, range_max))
    pd_gen, _ = np.histogram(generated_data, bins=bins, density=True, range=(range_min, range_max))
    kld = 0
    for x1, x2 in zip(pd_gt, pd_gen):
        if x1 != 0 and x2 == 0:
            kld += x1
        elif x1 == 0 and x2 != 0:
            kld += x2
        elif x1 != 0 and x2 != 0:
            kld += x1 * np.log(x1 / x2)

    return np.abs(kld)