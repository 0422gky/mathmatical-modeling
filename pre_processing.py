# 这个文件是关于数据预处理的函数封装，以ARIMA模型（2024C题）的数据预处理为框架
# 使用方法： from pre_processing.py import *   之后便可以在同一文件夹下的文件内调用改pre_processing的封装函数
# 主要想要在该文件中1实现的是异常值处理函数，函数的功能都可以实时扩展


# --------- 函数一：使用iqr和std标准差的异常值剔除函数 ---------------------------------------------------
# 定义异常值处理函数
def remove_outliers(data, column, method="std", factor=3):
    """
    函数参数解释：
    data    一般是通过read() 处理了csv,exls等文件格式得到的数据
    column  由于读取类型一般是表格，column代表我们读取的表格当中的各个列
    method:  
            std:   异常值剔除方法：std表示通过标准差剔除异常值,函数的默认参数就是std
            iqr:   异常值剔除方法：iqr方法，其中factor = 3是它的统计学意义，一般情况下不要更改这个factor = 3
    factor  用于剔除异常值的范围参数，std标准差方法剔除情况下，lower_bound = mean - factor * std，upper_bound = mean + factor * std
    """
    if method == "std":
        mean = data[column].mean()
        std = data[column].std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
    elif method == "iqr":
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
    else:
        raise ValueError("Unsupported method. Use 'std' or 'iqr'.")
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]




# --------------------------数据分组和汇总操作-----------------------------------
def group_and_summarize(data, group_cols, target_col, agg_func='sum'):
    """
    按指定列分组并汇总数据。
    
    参数:
        data (pd.DataFrame): 数据框。
        group_cols (list): 用于分组的列名列表。
        target_col (str): 需要汇总的目标列。
        agg_func (str): 汇总函数，默认为 'sum'。
    
    返回:
        pd.DataFrame: 分组汇总后的数据。
    """
    grouped_data = data.groupby(group_cols)[target_col].agg(agg_func).reset_index()
    return grouped_data

'''
使用实例
# 按日期和分拣中心分组，计算每日货量总和
data = group_and_summarize(data, group_cols=['Date', 'Center'], target_col='Quantity', agg_func='sum')
'''