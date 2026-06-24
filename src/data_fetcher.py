import os
import pandas as pd
import yfinance as yf
import akshare as ak

# 代理配置：仅在环境变量已设置时生效（本地开发按需配置，CI 环境无需代理）
# 若需本地代理，请在 .env 中设置 HTTP_PROXY / HTTPS_PROXY，不要硬编码

def fetch_us_soxx(start_date="2020-01-01"):
    """
    获取美股 SOXX 的日线数据，计算每日涨跌幅
    """
    print("Fetching US SOXX data...")
    soxx = yf.Ticker("SOXX")
    df = soxx.history(start=start_date)
    df = df[['Close']].copy()
    # 去除时区，对齐日期
    df.index = df.index.tz_localize(None).normalize()
    # 计算 T-1 相对于 T-2 的涨跌幅: (Close - Close(-1)) / Close(-1)
    df['SOXX_Return'] = df['Close'].pct_change()
    return df

def fetch_korea_samsung(start_date="2020-01-01"):
    """
    获取韩股 三星电子 (005930.KS) 日线数据，计算跳空涨跌幅
    """
    print("Fetching Korea Samsung data...")
    samsung = yf.Ticker("005930.KS")
    df = samsung.history(start=start_date)
    df = df[['Open', 'Close']].copy()
    df.index = df.index.tz_localize(None).normalize()
    # 计算 T 日开盘相对 T-1 收盘的跳空涨跌幅: (Open - Close(-1)) / Close(-1)
    df['Samsung_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    return df

def fetch_ashare_etf(start_date="20200101"):
    """
    获取 A 股半导体 ETF (512480) 日线数据，计算日内涨跌幅标签
    """
    print("Fetching A-share ETF data...")
    # 使用 akshare 获取 ETF 历史数据
    df = ak.fund_etf_hist_em(symbol="512480", period="daily", start_date=start_date, adjust="qfq")
    df = df[['日期', '开盘', '收盘']].copy()
    df.columns = ['Date', 'Open', 'Close']
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # 计算 T 日内表现: (Close - Open) / Open
    df['A_ETF_Intraday'] = (df['Close'] - df['Open']) / df['Open']
    df['A_ETF_Label'] = (df['A_ETF_Intraday'] > 0).astype(int)
    return df
