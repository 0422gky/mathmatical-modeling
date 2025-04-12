import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from matplotlib import rcParams
import matplotlib.font_manager as fm

# 指定宋体字体路径（从 data 文件夹加载）
font_path = "./data/SIMSUN.ttc"  # 确保路径正确
font_prop = fm.FontProperties(fname=font_path)

# 设置全局字体
rcParams['font.sans-serif'] = [font_prop.get_name()]  # 使用宋体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
    sns.set_theme(style="whitegrid", font="SIMSUN")

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
    plt.title('goods quantity change situation')
    plt.ylabel('quantity')
    plt.xlabel('date')

    # 调整 x 轴刻度倾斜角度
    plt.xticks(rotation=45)
    
    # 设置图例位置到右侧
    plt.legend(title='center',
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

    sns.lineplot(data=SC_data, x='Date', y='Quantity', label='before preprocessing', linewidth=1.5, color='blue')
    sns.lineplot(data=SC_cleaned, x='Date', y='Quantity', label='after preprocessing', linewidth=1.5, color='orange')

    plt.title(f'{SC} changing situation before and after preprocessing', fontsize=16)
    plt.ylabel('quantity', fontsize=12)
    plt.xlabel('date', fontsize=12)

    plt.xticks(rotation=45)

    plt.legend(title='data situation', fontsize=10, loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout()
    plt.show()

# 读取附件二
file_name2 = "./data/附件2.csv" 
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
    plt.title('quantity delta every hour', fontsize=16)
    plt.ylabel('increased quantity every hour', fontsize=12)
    plt.xlabel('hour', fontsize=12)

    # 优化布局
    plt.tight_layout()
    plt.show()
else:
    print("数据中缺少必要的列：'Hour', 'Quantity'")

#论文一前五页在这里做完了,描述性统计，目的是为了之后剔除数据做预处理的时候能有更好的思路

# 模型检测
def Model_checking(model) -> None:
    print('------------残差检验-----------')
    print(stats.normaltest(model.resid))
 
    plt.figure(figsize=(10, 6))
    qqplot(model.resid, line="q", fit=True)
    plt.title("Q-Qplot")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(model.resid, bins=50)
    plt.title("残差直方图")
    plt.show()
 
    jb_test = sm.stats.stattools.jarque_bera(model.resid)
    print("==================================================")
    print('------------Jarque-Bera检验-----------')
    print('JB:', jb_test[0])
    print('p-value:', jb_test[1])
    print('Skew:', jb_test[2])
    print('Kurtosis:', jb_test[3])
 
    print("==================================================")
    print('------DW检验:残差序列自相关----')
    print(sm.stats.stattools.durbin_watson(model.resid.values))


# 这里有一个问题，一个center的数据量在附件二里面实在是太多了，导致我们基本只画一个center的图就会导致电脑卡住，没法处理SC54后面的center
def predict_and_visualize(data2, center, forecast_days=30):
    """
    对指定分拣中心进行预测并可视化
    
    Parameters:
    -----------
    data2: DataFrame
        原始数据
    center: str
        分拣中心名称
    forecast_days: int
        预测天数
    """
    # 获取该分拣中心的数据
    center_data = data2[data2['Center'] == center].copy()
    
    # 将Date列转换为datetime类型
    center_data['Date'] = pd.to_datetime(center_data['Date'])
    
    # 获取最近30天的数据
    recent_dates = center_data['Date'].unique()[-30:]
    recent_data = center_data[center_data['Date'].isin(recent_dates)]
    
    # 创建预测结果DataFrame
    all_forecasts = pd.DataFrame()
    
    # 对每个小时分别进行时间序列分析和预测
    for hour in range(24):
        if hour in recent_data['Hour'].unique():
            print(f"\n分析 {hour}:00 的时间序列...")
            hour_data = recent_data[recent_data['Hour'] == hour].set_index('Date')['Quantity']
            
            # 时序数据平稳性检测
            original_ADF = ADF(hour_data)
            print(f"{hour}:00 的ADF检验结果为:", original_ADF)
            
            # 对数据进行差分运算，直到序列平稳
            diff_num = 0
            diff_data = hour_data.copy()
            ADF_p_value = ADF(diff_data)[1]
            
            while ADF_p_value > 0.01 and diff_num < 2:
                diff_data = diff_data.diff().dropna()
                diff_num += 1
                ADF_result = ADF(diff_data) # 有的center存在问题：ADF检验时样本量太小
                ADF_p_value = ADF_result[1]
                print(f"{hour}:00 的{diff_num}阶差分的ADF检验结果为:", ADF_result)
            
            # 使用AIC和BIC准则定阶
            max_ar = min(4, len(diff_data) // 4)
            max_ma = min(4, len(diff_data) // 4)
            
            try:
                order_select = sm.tsa.stattools.arma_order_select_ic(
                    diff_data, max_ar=max_ar, max_ma=max_ma, ic=['aic', 'bic'])
                p = order_select.bic_min_order[0]
                q = order_select.bic_min_order[1]
                print(f"{hour}:00 的ARIMA阶数为: p={p}, d={diff_num}, q={q}")
            except:
                print(f"{hour}:00 自动定阶失败，使用默认阶数 p=1, q=1")
                p, q = 1, 1
            
            # 构建ARIMA模型
            try:
                model = ARIMA(hour_data, order=(p, diff_num, q)).fit()
                forecast = model.forecast(steps=forecast_days)
                conf = model.get_forecast(steps=forecast_days).conf_int()
                
                # 创建预测结果DataFrame
                forecast_df = pd.DataFrame({
                    'Center': center,
                    'Hour': hour,
                    'Date': pd.date_range(start=hour_data.index[-1] + pd.Timedelta(days=1), 
                                        periods=forecast_days, freq='D'),
                    'Forecast': forecast,
                    'Lower_CI': conf.iloc[:, 0],
                    'Upper_CI': conf.iloc[:, 1]
                })
                
                # 合并所有预测结果
                all_forecasts = pd.concat([all_forecasts, forecast_df])
                
            except Exception as e:
                print(f"分拣中心 {center} 在 {hour}:00 的预测失败: {e}")
    
    # 创建历史数据DataFrame
    historical_data = recent_data[['Date', 'Hour', 'Quantity']].copy()
    historical_data['Type'] = '历史数据'
    
    # 创建预测数据DataFrame
    forecast_data = all_forecasts[['Date', 'Hour', 'Forecast']].copy()
    forecast_data = forecast_data.rename(columns={'Forecast': 'Quantity'})
    forecast_data['Type'] = '预测数据'
    
    # 合并历史数据和预测数据
    all_data = pd.concat([historical_data, forecast_data])
    
    # 创建时间索引
    all_data['DateTime'] = all_data.apply(lambda x: x['Date'] + pd.Timedelta(hours=x['Hour']), axis=1)
    all_data = all_data.sort_values('DateTime')
    
    # 创建图表
    plt.figure(figsize=(30, 10))
    
    # 绘制历史数据
    plt.plot(range(len(all_data[all_data['Type'] == '历史数据'])), 
            all_data[all_data['Type'] == '历史数据']['Quantity'], 
            'b-', label='历史数据', alpha=0.7)
    
    # 绘制预测数据
    plt.plot(range(len(all_data[all_data['Type'] == '历史数据']), 
                  len(all_data)), 
            all_data[all_data['Type'] == '预测数据']['Quantity'], 
            'r--', label='预测数据', alpha=0.7)
    
    # 绘制置信区间
    plt.fill_between(range(len(all_data[all_data['Type'] == '历史数据']), 
                          len(all_data)),
                    all_forecasts['Lower_CI'],
                    all_forecasts['Upper_CI'],
                    color='gray', alpha=0.2, label='95%置信区间')
    
    # 设置图表属性
    plt.title(f'分拣中心 {center} 历史30天和未来30天每小时货量预测', fontsize=14)
    plt.xlabel('时间点', fontsize=12)
    plt.ylabel('货量', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴刻度
    total_points = len(all_data)
    plt.xticks(range(0, total_points, 24), 
              [f'第{i//24+1}天' for i in range(0, total_points, 24)])
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f'forecast_{center}_all_hours.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存预测结果到CSV
    all_forecasts.to_csv(f'forecast_{center}_all_hours.csv', index=False)
    print(f"分拣中心 {center} 的预测结果已保存到 forecast_{center}_all_hours.csv")

# 主程序
if {'Hour', 'Quantity', 'Center', 'Date'}.issubset(data2.columns):
    # 获取所有分拣中心
    centers = data2['Center'].unique()
    
    # 对每个分拣中心进行预测和可视化
    for center in centers:
        print(f"\n开始分析分拣中心 {center} 的时间序列...")
        predict_and_visualize(data2, center)
else:
    print("数据中缺少必要的列：'Hour', 'Quantity', 'Center', 'Date'")