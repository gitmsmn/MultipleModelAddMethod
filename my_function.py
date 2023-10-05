# ライブラリ
## 基本
import numpy as np
import matplotlib.pyplot as plt

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
def calc_mse(y_pred, y_true):
    """
    MSEを計算する

    """
    return np.mean((y_true - y_pred) ** 2)

def calc_rmse(y_pred, y_true):
    """
    RMSEを計算する

    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

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