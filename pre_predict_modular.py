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

# 导入自定义模块
from time_series_utils import TimeSeriesAnalyzer
from visualization_utils import TimeSeriesVisualizer
from pre_processing import remove_outliers, group_and_summarize
from read_file import read_file

class DataAnalyzer:
    """数据分析类，负责数据的加载、预处理和基本分析"""
    
    def __init__(self, font_path="./data/SIMSUN.ttc"):
        """
        初始化数据分析类
        
        Parameters:
        -----------
        font_path: str
            字体文件路径
        """
        self.font_path = font_path
        self.setup_visualization()
        self.visualizer = TimeSeriesVisualizer(font_path)
    
    def setup_visualization(self):
        """设置可视化环境"""
        font_prop = fm.FontProperties(fname=self.font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style="whitegrid", font="SIMSUN")
    
    def load_and_preprocess_data(self, file_path, file_type='csv', **kwargs):
        """
        加载并预处理数据
        
        Parameters:
        -----------
        file_path: str
            文件路径
        file_type: str
            文件类型
        **kwargs: dict
            传递给read_file的其他参数
            
        Returns:
        --------
        pd.DataFrame: 处理后的数据
        """
        data = read_file(file_path, file_type=file_type, **kwargs)
        if data is not None:
            # 确保日期列是datetime类型
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            return data
        return None
    
    def plot_quantity_trend(self, data, x_col='Date', y_col='Quantity', hue_col='Center'):
        """
        绘制货量趋势图
        
        Parameters:
        -----------
        data: pd.DataFrame
            数据
        x_col: str
            x轴列名
        y_col: str
            y轴列名
        hue_col: str
            分组列名
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x=x_col, y=y_col, hue=hue_col, linewidth=0.8)
        plt.title('Goods Quantity Change Situation')
        plt.ylabel('Quantity')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.legend(title='Center', fontsize=5, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
    
    def plot_outlier_comparison(self, data, center, column='Quantity', method="iqr", factor=1.5):
        """
        绘制异常值处理前后的对比图
        
        Parameters:
        -----------
        data: pd.DataFrame
            原始数据
        center: str
            分拣中心名称
        column: str
            目标列名
        method: str
            异常值处理方法
        factor: float
            异常值处理因子
        """
        center_data = data[data['Center'] == center]
        cleaned_data = remove_outliers(center_data, column, method, factor)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=center_data, x='Date', y=column, 
                    label='Before Preprocessing', linewidth=1.5, color='blue')
        sns.lineplot(data=cleaned_data, x='Date', y=column, 
                    label='After Preprocessing', linewidth=1.5, color='orange')
        
        plt.title(f'{center} Changing Situation Before and After Preprocessing', fontsize=16)
        plt.ylabel('Quantity', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Data Situation', fontsize=10, 
                  loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.show()

class TimeSeriesPredictor:
    """时间序列预测类，负责ARIMA模型预测"""
    
    def __init__(self, data_analyzer):
        """
        初始化时间序列预测类
        
        Parameters:
        -----------
        data_analyzer: DataAnalyzer
            数据分析类实例
        """
        self.data_analyzer = data_analyzer
        self.analyzer = None  # 将在predict_center方法中初始化
    
    def check_stationarity(self, series, max_diff=2, significance_level=0.01):
        """
        检查时间序列的平稳性并进行差分
        
        Parameters:
        -----------
        series: pd.Series
            时间序列数据
        max_diff: int
            最大差分次数
        significance_level: float
            显著性水平
            
        Returns:
        --------
        tuple: (平稳化后的数据, 差分次数)
        """
        diff_num = 0
        diff_data = series.copy()
        
        while diff_num < max_diff:
            try:
                adf_result = ADF(diff_data)
                if adf_result[1] <= significance_level:
                    break
                diff_data = diff_data.diff().dropna()
                diff_num += 1
            except Exception as e:
                print(f"ADF检验失败: {e}")
                break
                
        return diff_data, diff_num
    
    def select_arima_order(self, data, max_ar=4, max_ma=4):
        """
        使用AIC和BIC准则选择ARIMA模型的最优阶数
        
        Parameters:
        -----------
        data: pd.Series
            时间序列数据
        max_ar: int
            最大自回归阶数
        max_ma: int
            最大移动平均阶数
            
        Returns:
        --------
        tuple: (p, q)
        """
        try:
            order_select = sm.tsa.stattools.arma_order_select_ic(
                data, max_ar=max_ar, max_ma=max_ma, ic=['aic', 'bic'])
            p = order_select.bic_min_order[0]
            q = order_select.bic_min_order[1]
            return p, q
        except:
            print("自动定阶失败，使用默认阶数 p=1, q=1")
            return 1, 1
    
    def model_diagnostics(self, model):
        """
        进行模型诊断
        
        Parameters:
        -----------
        model: ARIMAResults
            ARIMA模型拟合结果
        """
        print('------------残差检验-----------')
        print(stats.normaltest(model.resid))
        
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
    
    def predict_center(self, data, center, forecast_days=30):
        """
        对指定分拣中心进行预测
        
        Parameters:
        -----------
        data: pd.DataFrame
            原始数据
        center: str
            分拣中心名称
        forecast_days: int
            预测天数
            
        Returns:
        --------
        tuple: (预测结果DataFrame, 历史数据DataFrame)
        """
        # 获取该分拣中心的数据
        center_data = data[data['Center'] == center].copy()
        
        # 将Date列转换为datetime类型（如果还不是）
        if not pd.api.types.is_datetime64_any_dtype(center_data['Date']):
            center_data['Date'] = pd.to_datetime(center_data['Date'])
        
        # 获取最近30天的数据
        recent_dates = sorted(center_data['Date'].unique())[-30:]  # 确保只取最近30天
        recent_data = center_data[center_data['Date'].isin(recent_dates)].copy()
        
        # 创建完整的30天24小时时间索引
        full_dates = pd.date_range(start=recent_dates[0], end=recent_dates[-1], freq='D')
        full_hours = range(24)
        
        # 创建完整的历史数据DataFrame
        historical_data = pd.DataFrame()
        for date in full_dates:
            for hour in full_hours:
                temp_data = recent_data[
                    (recent_data['Date'] == date) & 
                    (recent_data['Hour'] == hour)
                ]
                if len(temp_data) == 0:
                    # 如果某个时间点没有数据，填充0
                    historical_data = pd.concat([
                        historical_data,
                        pd.DataFrame({
                            'Date': [date],
                            'Hour': [hour],
                            'Quantity': [0]
                        })
                    ], ignore_index=True)
                else:
                    historical_data = pd.concat([
                        historical_data,
                        temp_data[['Date', 'Hour', 'Quantity']]
                    ], ignore_index=True)
        
        # 确保历史数据按日期和小时排序
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        historical_data = historical_data.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        # 创建预测结果DataFrame
        all_forecasts = pd.DataFrame()
        
        # 对每个小时分别进行预测
        for hour in range(24):
            print(f"\n分析 {hour}:00 的时间序列...")
            
            # 获取当前小时的数据
            hour_data = historical_data[historical_data['Hour'] == hour].set_index('Date')['Quantity']
            
            # 如果这个小时没有数据，创建一个空的Series
            if len(hour_data) == 0:
                print(f"{hour}:00 没有历史数据，使用0值填充")
                last_date = recent_dates[-1]
                
                dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days
                )
                forecast = pd.Series([0] * forecast_days, index=dates)
                conf = pd.DataFrame({
                    0: [0] * forecast_days,
                    1: [0] * forecast_days
                }, index=dates)
            else:
                # 检查数据量是否足够
                if len(hour_data) < 10:
                    print(f"数据点不足，缺失部分填充0")
                    dates = pd.date_range(
                        start=recent_dates[-1] + pd.Timedelta(days=1), 
                        periods=forecast_days
                    )
                    
                    # 生成全0序列
                    forecast = pd.Series([0] * forecast_days, index=dates)
                    conf = pd.DataFrame({
                        0: [0] * forecast_days,
                        1: [0] * forecast_days
                    }, index=dates)
                else:
                    # 检查是否是常量序列
                    if hour_data.nunique() <= 1:
                        print(f"{hour}:00 的数据是常量，使用常量值进行预测")
                        constant_value = hour_data.iloc[0]
                        dates = pd.date_range(
                            start=recent_dates[-1] + pd.Timedelta(days=1), 
                            periods=forecast_days
                        )
                        forecast = pd.Series([constant_value] * forecast_days, index=dates)
                        conf = pd.DataFrame({
                            0: [constant_value] * forecast_days,
                            1: [constant_value] * forecast_days
                        }, index=dates)
                    else:
                        # 初始化TimeSeriesAnalyzer
                        if self.analyzer is None:
                            self.analyzer = TimeSeriesAnalyzer(data=recent_data)
                        
                        # 进行预测
                        forecast, conf = self.analyzer.predict_hourly_data(hour_data, forecast_days)
                        
                        if forecast is None:
                            # 如果预测失败，使用0值填充
                            dates = pd.date_range(
                                start=recent_dates[-1] + pd.Timedelta(days=1), 
                                periods=forecast_days
                            )
                            forecast = pd.Series([0] * forecast_days, index=dates)
                            conf = pd.DataFrame({
                                0: [0] * forecast_days,
                                1: [0] * forecast_days
                            }, index=dates)
            
            # 创建预测结果DataFrame
            forecast_df = pd.DataFrame({
                'Center': center,
                'Hour': hour,
                'Date': forecast.index,
                'Forecast': forecast.values,
                'Lower_CI': conf.iloc[:, 0].values,
                'Upper_CI': conf.iloc[:, 1].values
            })
            
            all_forecasts = pd.concat([all_forecasts, forecast_df], ignore_index=True)
        
        # 确保预测结果按日期和小时排序
        all_forecasts['Date'] = pd.to_datetime(all_forecasts['Date'])
        all_forecasts = all_forecasts.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        return all_forecasts, historical_data

def main():
    """主函数"""
    # 初始化数据分析类
    analyzer = DataAnalyzer()
    
    # 加载数据
    data1 = analyzer.load_and_preprocess_data(
        "./data/附件1.csv",
        parse_dates=['Date']
    )
    data2 = analyzer.load_and_preprocess_data(
        "./data/附件2.csv",
        parse_dates=['Date']
    )
    
    if data1 is not None and data2 is not None:
        # 预处理数据
        data1 = group_and_summarize(data1, ['Date', 'Center'], 'Quantity')
        
        # 绘制整体趋势图
        analyzer.plot_quantity_trend(data1)
        
        # 绘制异常值处理对比图
        for center in ["SC20", "SC21"]:
            analyzer.plot_outlier_comparison(data1, center)
        
        # 初始化预测类
        predictor = TimeSeriesPredictor(analyzer)
        
        # 对每个分拣中心进行预测和可视化
        centers = data2['Center'].unique()
        for center in centers:
            print(f"\n开始分析分拣中心 {center} 的时间序列...")
            forecast_data, historical_data = predictor.predict_center(data2, center)
            
            if not forecast_data.empty:
                # 可视化预测结果
                analyzer.visualizer.plot_forecast(
                    historical_data=historical_data,
                    forecast_data=forecast_data,
                    center=center,
                    save_path=f'forecast_{center}_all_hours.png'
                )
                # 保存预测结果
                forecast_data.to_csv(f'forecast_{center}_all_hours.csv', index=False)
                print(f"分拣中心 {center} 的预测结果已保存")

if __name__ == "__main__":
    main() 