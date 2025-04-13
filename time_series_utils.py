# time_series_utils.py
# 采用ARIMA模型的时间序列分析
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

class TimeSeriesAnalyzer:
    def __init__(self, data=None, date_col='Date', value_col='Quantity'):
        """
        初始化时间序列分析器
        
        Parameters:
        -----------
        data: pd.DataFrame
            包含时间序列数据的DataFrame
        date_col: str
            日期列名
        value_col: str
            数值列名
        """
        self.data = data.copy() if data is not None else None
        self.date_col = date_col
        self.value_col = value_col
        if self.data is not None and date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        
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
    
    def predict_hourly_data(self, hour_data, forecast_days=30):
        """
        对单个小时的数据进行预测
        
        Parameters:
        -----------
        hour_data: pd.Series
            单个小时的时间序列数据
        forecast_days: int
            预测天数
            
        Returns:
        --------
        tuple: (预测结果, 置信区间)
        """
        try:
            # 数据预处理
            if len(hour_data) < 10:
                print("数据点不足，缺失部分填充0")
                # 生成完整的日期序列
                dates = pd.date_range(
                    start=hour_data.index[-1] + pd.Timedelta(days=1), 
                    periods=forecast_days
                )
                
                # 生成全0序列
                forecast = pd.Series([0] * forecast_days, index=dates)
                conf = pd.DataFrame({
                    0: [0] * forecast_days,
                    1: [0] * forecast_days
                }, index=dates)
                return forecast, conf

            # 检查数据是否为常量
            if hour_data.nunique() <= 1:
                print("数据为常量序列，使用常量值进行预测")
                constant_value = hour_data.iloc[0]
                dates = pd.date_range(
                    start=hour_data.index[-1] + pd.Timedelta(days=1), 
                    periods=forecast_days
                )
                forecast = pd.Series([constant_value] * forecast_days, index=dates)
                conf = pd.DataFrame({
                    0: [constant_value] * forecast_days,
                    1: [constant_value] * forecast_days
                }, index=dates)
                return forecast, conf

            # 数据标准化
            mean_val = hour_data.mean()
            std_val = hour_data.std()
            if std_val == 0:
                std_val = 1
            normalized_data = (hour_data - mean_val) / std_val

            # 检查数据平稳性并进行差分
            diff_data, diff_num = self.check_stationarity(normalized_data)
            
            # 选择ARIMA模型阶数
            max_ar = min(3, len(diff_data) // 5)
            max_ma = min(3, len(diff_data) // 5)
            p, q = self.select_arima_order(diff_data, max_ar=max_ar, max_ma=max_ma)
            
            try:
                # 尝试拟合ARIMA模型
                model = ARIMA(normalized_data, order=(p, diff_num, q)).fit()
            except:
                # 如果拟合失败，尝试更简单的模型
                print("ARIMA拟合失败，尝试更简单的模型")
                model = ARIMA(normalized_data, order=(1, diff_num, 0)).fit()

            # 进行预测
            normalized_forecast = model.forecast(steps=forecast_days)
            normalized_conf = model.get_forecast(steps=forecast_days).conf_int()

            # 反标准化
            forecast = normalized_forecast * std_val + mean_val
            conf = normalized_conf * std_val + mean_val

            # 确保预测结果长度正确
            if len(forecast) != forecast_days:
                print(f"警告：预测结果长度({len(forecast)})与预期天数({forecast_days})不符")
                dates = pd.date_range(
                    start=hour_data.index[-1] + pd.Timedelta(days=1), 
                    periods=forecast_days
                )
                
                if len(forecast) < forecast_days:
                    # 如果预测结果不足，用0填充剩余部分
                    padding = pd.Series([0] * (forecast_days - len(forecast)),
                                     index=dates[len(forecast):])
                    forecast = pd.concat([forecast, padding])
                    
                    # 同样填充置信区间
                    padding_conf = pd.DataFrame({
                        0: [0] * (forecast_days - len(conf)),
                        1: [0] * (forecast_days - len(conf))
                    }, index=dates[len(conf):])
                    conf = pd.concat([conf, padding_conf])
                else:
                    # 如果预测结果过长，截取所需部分
                    forecast = forecast[:forecast_days]
                    conf = conf[:forecast_days]
                
                # 确保索引正确
                forecast.index = dates
                conf.index = dates

            return forecast, conf

        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            # 发生错误时使用0值填充
            dates = pd.date_range(
                start=hour_data.index[-1] + pd.Timedelta(days=1), 
                periods=forecast_days
            )
            forecast = pd.Series([0] * forecast_days, index=dates)
            conf = pd.DataFrame({
                0: [0] * forecast_days,
                1: [0] * forecast_days
            }, index=dates)
            return forecast, conf 