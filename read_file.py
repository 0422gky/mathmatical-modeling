#------------------------读取文件函数----------------------------------------------------
# 该函数用于文件的读取，目前编写的这个用于csv和excel文件的读取，默认的读取文件格式是csv文件

import pandas as pd

def read_file(file_path, file_type='csv', dropna=True, usecols=None, parse_dates=None):
    """
    通用文件读取函数，用于读取 CSV 或 Excel 文件并进行预处理。
    
    参数:
        file_path (str): 文件路径。
        file_type (str): 文件类型，支持 'csv' 或 'excel'。
        dropna (bool): 是否删除缺失值，默认为 True。
        usecols (list): 指定需要读取的列，例如 ['column1', 'column2']。
        parse_dates (list): 指定需要解析为日期的列，例如 ['date_column']。
    
    返回:
        pd.DataFrame: 预处理后的数据框。
    """
    try:
        # 根据文件类型读取文件
        if file_type == 'csv':
            data = pd.read_csv(file_path, usecols=usecols, parse_dates=parse_dates)
        elif file_type == 'excel':
            data = pd.read_excel(file_path, usecols=usecols, parse_dates=parse_dates)
        else:
            raise ValueError("不支持的文件类型，仅支持 'csv' 和 'excel'")
        
        # 删除缺失值
        if dropna:
            data = data.dropna()
        
        print(f"文件 {file_path} 读取成功，数据维度为: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径是否正确。")
    except ValueError as ve:
        print(f"文件类型错误: {ve}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

'''
假设你的文件路径为 ./data/附件2.csv 可以这样调用
from read_file import read_file

# 读取文件并进行预处理
file_path = "./data/附件2.csv"
data = read_file(
    file_path,
    file_type='csv',
    dropna=True,
    usecols=['Center', 'Date', 'Hour', 'Quantity'],
    parse_dates=['Date']
)

# 查看数据
print(data.head())
'''