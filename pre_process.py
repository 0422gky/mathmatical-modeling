import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取附件一
file_name = "./data/附件1.csv" 
try:
    data = pd.read_csv(file_name,).dropna()
    print(data.head())
except Exception as e:
    print(f"读取文件时出错: {e}")

# 定义异常值处理函数
def remove_outliers(data, column, method="std", factor=3):
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

# 检查数据是否加载成功
if {'Date', 'Center', 'Quantity'}.issubset(data.columns):

    # 将日期列转换为日期格式
    data['Date'] = pd.to_datetime(data['Date'])

    # 检查是否需要分组
    if data.duplicated(subset=['Date', 'Center']).any():
        # 按日期和分拣中心分组，计算每日货量总和
        data = data.groupby(['Date', 'Center'])['Quantity'].sum().reset_index()
    else:
        # 数据已汇总，直接使用
        data = data

    # 设置 Seaborn 全局样式
    sns.set_theme(style="whitegrid", font="SimSong")

    # 动态调整图形大小
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data, 
        x='Date', 
        y='Quantity', 
        hue='Center', 
        linewidth=0.8
    )

    # 设置标题和坐标轴标签
    plt.title('各分拣中心日货量变化情况')
    plt.ylabel('货量')
    plt.xlabel('日期')

    # 调整 x 轴刻度倾斜角度
    plt.xticks(rotation=45)
    
    # 设置图例位置到右侧
    plt.legend(title='分拣中心',
               fontsize=5,
               bbox_to_anchor=(1.05, 1))
    
    # 优化布局
    plt.tight_layout()
    plt.show()
else:
    print("数据中缺少必要的列：'Date', 'Center', 'Quantity'")

# 添加 SC20 的剔除前后对比
center_name = ["SC20", "SC21"]
for SC in center_name:
    SC_data = data[data['Center'] == SC]

    # 剔除异常值（使用三倍标准差法）
    SC_cleaned = remove_outliers(SC_data, column='Quantity', method="iqr", factor=1.5)

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=SC_data, x='Date', y='Quantity', label='剔除前', linewidth=1.5, color='blue')
    sns.lineplot(data=SC_cleaned, x='Date', y='Quantity', label='剔除后', linewidth=1.5, color='orange')

    plt.title(f'{SC} 剔除前后货量变化情况', fontsize=16)
    plt.ylabel('货量', fontsize=12)
    plt.xlabel('日期', fontsize=12)

    plt.xticks(rotation=45)

    plt.legend(title='数据状态', fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    plt.show()

# 读取附件二
file_name2 = "附件2.csv" 
try:
    data2 = pd.read_csv(file_name2).dropna()
    print(data2.head())
except Exception as e:
    print(f"读取文件时出错: {e}")

# 按 Hour 分组，计算所有中心和所有天数的总货量
if {'Hour', 'Quantity'}.issubset(data2.columns):
    hourly_total = data2.groupby('Hour')['Quantity'].sum().reset_index()

    # 动态调整图形大小
    plt.figure(figsize=(12, 6))

    # 绘制折线图
    sns.lineplot(
        data = hourly_total,
        x = 'Hour',
        y = 'Quantity',
        linewidth = 1.5,
        color = 'blue',
        marker = "o"
    )

    # 设置标题和坐标轴标签
    plt.title('各小时总货量变化情况', fontsize=16)
    plt.ylabel('每小时总新增货量', fontsize=12)
    plt.xlabel('小时', fontsize=12)

    # 优化布局
    plt.tight_layout()
    plt.show()
else:
    print("数据中缺少必要的列：'Hour', 'Quantity'")

