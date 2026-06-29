"""
data_fetcher.py — 多市场数据获取模块
- 美股：yfinance 批量下载，取T-1日收益率作为预测因子
- 韩股：yfinance 批量下载，取T日开盘跳空作为预测因子
- A股：使用 yfinance 获取（添加 .SS 或 .SZ 后缀），避免 akshare 的网络不稳定问题
"""

import os
import pandas as pd
import yfinance as yf

from src.stocks_universe import get_all_us_tickers, get_all_kr_tickers, get_all_ashare_codes

def _parse_multiindex(raw: pd.DataFrame, field: str, tickers: list) -> pd.DataFrame:
    """从 yfinance 返回的 DataFrame 中提取指定 field（兼容单/多ticker）"""
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()

    if len(tickers) == 1:
        col = raw[field] if field in raw.columns else raw.iloc[:, 0]
        return col.to_frame(name=tickers[0])

    try:
        result = raw[field]
        if isinstance(result, pd.DataFrame):
            return result
    except (KeyError, TypeError):
        pass
    try:
        return raw.xs(field, level=0, axis=1)
    except Exception:
        pass
    try:
        return raw.xs(field, level=1, axis=1)
    except Exception:
        pass

    raise ValueError(f"Cannot extract '{field}' from columns: {list(raw.columns)[:10]}")


def fetch_us_stocks(start_date: str = "2025-09-01") -> pd.DataFrame:
    """
    批量获取所有美股因子 → 返回每日收益率 DataFrame
    """
    tickers = get_all_us_tickers()
    print(f"  Fetching {len(tickers)} US tickers: {tickers}")

    raw = yf.download(
        tickers if len(tickers) > 1 else tickers[0],
        start=start_date, progress=False, auto_adjust=True
    )
    close = _parse_multiindex(raw, "Close", tickers)
    returns = close.pct_change()
    returns.columns = [f"US_{t}" for t in returns.columns]
    print(f"  US done — {len(returns)} rows, last: {returns.index[-1].date()}")
    return returns


def fetch_korea_stocks(start_date: str = "2025-09-01") -> pd.DataFrame:
    """
    批量获取所有韩股因子 → 返回 (T日早盘10:00价格 - T-1日收盘价) / T-1日收盘价 DataFrame
    """
    tickers = get_all_kr_tickers()
    print(f"  Fetching {len(tickers)} Korea tickers: {tickers}")

    # 1. 获取日线数据以获得 T-1 日的收盘价
    daily_raw = yf.download(
        tickers if len(tickers) > 1 else tickers[0],
        start=start_date, progress=False, auto_adjust=True
    )
    daily_close = _parse_multiindex(daily_raw, "Close", tickers)

    # 2. 获取小时线数据以获得 10:00 的价格 (9:00 - 10:00 这根 K 线的 Close)
    hourly_raw = yf.download(
        tickers if len(tickers) > 1 else tickers[0],
        start=start_date, interval="1h", progress=False, auto_adjust=True
    )
    
    # 转换时区到韩国时间 (Asia/Seoul)
    if hourly_raw.index.tz is None:
        hourly_raw.index = hourly_raw.index.tz_localize('UTC').tz_convert('Asia/Seoul')
    else:
        hourly_raw.index = hourly_raw.index.tz_convert('Asia/Seoul')
        
    # 筛选出韩国时间 09:00 开始的那根小时线 (对应北京时间 08:00 - 09:00，刚好在 A 股盘前)
    hourly_09 = hourly_raw[hourly_raw.index.hour == 9].copy()
    
    # 将时间戳去掉时区并对齐到 00:00，以便和日线数据的日期对齐
    hourly_09.index = hourly_09.index.normalize().tz_localize(None)
    
    price_1000 = _parse_multiindex(hourly_09, "Close", tickers)

    # 3. 对齐日期并计算跳空
    idx = daily_close.index.intersection(price_1000.index)
    daily_close_aligned = daily_close.loc[idx]
    price_1000_aligned = price_1000.loc[idx]

    # 跳空收益率：(Price_1000_T - Close_{T-1}) / Close_{T-1}
    gaps = (price_1000_aligned - daily_close_aligned.shift(1)) / daily_close_aligned.shift(1)
    gaps.columns = [f"KR_{t.split('.')[0]}" for t in gaps.columns]
    print(f"  Korea (10:00 AM) done — {len(gaps)} rows, last: {gaps.index[-1].date() if len(gaps)>0 else 'None'}")
    return gaps


def fetch_ashare_stocks(start_date: str = "2025-09-01") -> dict:
    """
    使用 yfinance 获取A股日线数据。
    返回: {code: {'return': Series, 'label': Series, 'name': str}}
    """
    codes = get_all_ashare_codes()
    results = {}
    
    # 构造 yfinance 的 ticker 列表
    # 6开头的是上海(.SS), 0和3开头的是深圳(.SZ)
    yf_tickers = []
    ticker_to_code = {}
    for code in codes:
        suffix = ".SS" if code.startswith("6") else ".SZ"
        ticker = f"{code}{suffix}"
        yf_tickers.append(ticker)
        ticker_to_code[ticker] = code

    print(f"  Fetching {len(yf_tickers)} A-share tickers via yfinance...")
    raw = yf.download(
        yf_tickers if len(yf_tickers) > 1 else yf_tickers[0],
        start=start_date, progress=False, auto_adjust=True
    )
    
    closes = _parse_multiindex(raw, "Close", yf_tickers)
    opens = _parse_multiindex(raw, "Open", yf_tickers)

    for ticker in yf_tickers:
        code = ticker_to_code[ticker]
        name = codes[code]
        try:
            close_s = closes[ticker].dropna()
            open_s = opens[ticker].dropna()
            
            # 取交集对齐
            idx = close_s.index.intersection(open_s.index)
            if len(idx) == 0:
                print(f"    WARN: {name} ({code}) no data")
                continue
                
            close_s = close_s.loc[idx]
            open_s = open_s.loc[idx]

            intraday = (close_s - open_s) / open_s
            results[code] = {
                "return": intraday,
                "label": (intraday > 0).astype(int),
                "name": name,
            }
        except Exception as e:
            print(f"    WARN: {name} ({code}) failed — {e}")

    print(f"  A-share done — {len(results)}/{len(codes)} fetched OK")
    return results
