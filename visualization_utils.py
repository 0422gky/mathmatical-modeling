# visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TimeSeriesVisualizer:
    def __init__(self, font_path=None):
        """
        初始化时间序列可视化器
        
        Parameters:
        -----------
        font_path: str
            字体文件路径
        """
        if font_path:
            import matplotlib.font_manager as fm
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style="whitegrid")
    
    def plot_forecast(self, historical_data, forecast_data, center, save_path=None):
        """
        绘制预测结果，保持历史数据和预测数据的1:1比例
        
        Parameters:
        -----------
        historical_data: pd.DataFrame
            历史数据，包含Date, Hour, Quantity列
        forecast_data: pd.DataFrame
            预测数据，包含Date, Hour, Forecast, Lower_CI, Upper_CI列
        center: str
            分拣中心名称
        save_path: str
            保存路径
        """
        # 创建时间索引
        historical_data['DateTime'] = historical_data.apply(
            lambda x: pd.Timestamp(x['Date']) + pd.Timedelta(hours=x['Hour']), 
            axis=1
        )
        forecast_data['DateTime'] = forecast_data.apply(
            lambda x: pd.Timestamp(x['Date']) + pd.Timedelta(hours=x['Hour']), 
            axis=1
        )
        
        # 合并数据并按时间排序
        all_data = pd.concat([
            historical_data[['DateTime', 'Quantity']].rename(columns={'Quantity': 'Value'}),
            forecast_data[['DateTime', 'Forecast']].rename(columns={'Forecast': 'Value'})
        ]).sort_values('DateTime')
        
        # 创建图表，设置更宽的图形以保持1:1比例
        plt.figure(figsize=(20, 8))
        
        # 计算历史数据和预测数据的点数
        hist_points = len(historical_data)
        forecast_points = len(forecast_data)
        
        # 绘制历史数据
        hist_mask = all_data['DateTime'] <= historical_data['DateTime'].max()
        hist_data = all_data[hist_mask]
        plt.plot(range(len(hist_data)), 
                hist_data['Value'], 
                'b-', label='历史数据', alpha=0.7, linewidth=1.5)
        
        # 绘制预测数据
        forecast_mask = all_data['DateTime'] > historical_data['DateTime'].max()
        forecast_data_plot = all_data[forecast_mask]
        plt.plot(range(len(hist_data), len(all_data)), 
                forecast_data_plot['Value'], 
                'r--', label='预测数据', alpha=0.7, linewidth=1.5)
        
        # 绘制置信区间
        forecast_dates = forecast_data['DateTime'].sort_values()
        forecast_points = range(len(hist_data), len(all_data))
        
        # 确保置信区间数据与预测点数量一致
        lower_ci = forecast_data.set_index('DateTime').loc[forecast_dates, 'Lower_CI'].values
        upper_ci = forecast_data.set_index('DateTime').loc[forecast_dates, 'Upper_CI'].values
        
        if len(forecast_points) == len(lower_ci) == len(upper_ci):
            plt.fill_between(forecast_points,
                           lower_ci,
                           upper_ci,
                           color='gray', alpha=0.2, label='95%置信区间')
        else:
            print("警告：置信区间数据维度不匹配，跳过置信区间绘制")
        
        # 设置图表属性
        plt.title(f'分拣中心 {center} 历史30天和未来30天每小时货量预测', fontsize=14)
        plt.xlabel('时间点', fontsize=12)
        plt.ylabel('货量', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴刻度，每24小时显示一天
        total_points = len(all_data)
        x_ticks = list(range(0, total_points, 24))
        x_labels = [f'第{i//24+1}天' for i in range(0, total_points, 24)]
        plt.xticks(x_ticks, x_labels, rotation=45)
        
        # 添加垂直线标记历史数据和预测数据的分界
        plt.axvline(x=len(hist_data)-1, color='k', linestyle='--', alpha=0.5)
        
        # 保存或显示图表
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()