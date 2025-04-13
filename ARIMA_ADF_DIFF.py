# 实现ADF检验和差分操作的封装
from statsmodels.tsa.stattools import adfuller as ADF
def check_stationarity_and_diff(data, max_diff=2, significance_level=0.01):
    """
    检测时间序列的平稳性，并进行差分直到序列平稳。
    
    参数:
        data (pd.Series): 时间序列数据。
        max_diff (int): 最大差分次数，默认为 2。
        significance_level (float): ADF 检验的显著性水平，默认为 0.01。
    
    返回:
        pd.Series: 平稳化后的时间序列。
        int: 差分次数。
    """
    diff_num = 0
    while diff_num < max_diff:
        adf_p_value = ADF(data)[1]
        if adf_p_value <= significance_level:
            break
        data = data.diff().dropna()
        diff_num += 1
    return data, diff_num