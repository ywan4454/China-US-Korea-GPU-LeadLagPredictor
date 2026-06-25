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


def fetch_us_stocks(start_date: str = "2022-01-01") -> pd.DataFrame:
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


def fetch_korea_stocks(start_date: str = "2022-01-01") -> pd.DataFrame:
    """
    批量获取所有韩股因子 → 返回T日开盘跳空收益率 DataFrame
    """
    tickers = get_all_kr_tickers()
    print(f"  Fetching {len(tickers)} Korea tickers: {tickers}")

    raw = yf.download(
        tickers if len(tickers) > 1 else tickers[0],
        start=start_date, progress=False, auto_adjust=True
    )
    close = _parse_multiindex(raw, "Close", tickers)
    open_px = _parse_multiindex(raw, "Open", tickers)

    # 跳空收益率：(Open_T - Close_{T-1}) / Close_{T-1}
    gaps = (open_px - close.shift(1)) / close.shift(1)
    gaps.columns = [f"KR_{t.split('.')[0]}" for t in gaps.columns]
    print(f"  Korea done — {len(gaps)} rows, last: {gaps.index[-1].date()}")
    return gaps


def fetch_ashare_stocks(start_date: str = "2022-01-01") -> dict:
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
