"""
data_fetcher.py — 多市场数据获取模块
- 美股：yfinance 批量下载，取T-1日收益率作为预测因子
- 韩股：yfinance 批量下载，取T日开盘跳空作为预测因子
- A股：akshare 逐只获取日线数据（支持国内直连）
"""

import os
import time
import contextlib
import pandas as pd
import yfinance as yf
import akshare as ak

from src.stocks_universe import get_all_us_tickers, get_all_kr_tickers, get_all_ashare_codes


@contextlib.contextmanager
def _no_proxy():
    """
    临时设置 NO_PROXY=* 让 akshare 绕过所有代理直连国内域名。
    requests 库优先检查 NO_PROXY；设为 * 则对所有 host 绕过代理。
    退出上下文后恢复原始 NO_PROXY 值，yfinance 的代理设置不受影响。
    """
    saved_no_proxy = os.environ.get("NO_PROXY")
    saved_no_proxy_lower = os.environ.get("no_proxy")
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        if saved_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = saved_no_proxy
        if saved_no_proxy_lower is None:
            os.environ.pop("no_proxy", None)
        else:
            os.environ["no_proxy"] = saved_no_proxy_lower


def _parse_multiindex(raw: pd.DataFrame, field: str, tickers: list) -> pd.DataFrame:
    """从 yfinance 返回的 DataFrame 中提取指定 field（兼容单/多ticker）"""
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()

    if len(tickers) == 1:
        col = raw[field] if field in raw.columns else raw.iloc[:, 0]
        return col.to_frame(name=tickers[0])

    # 多ticker：MultiIndex columns
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
    列名格式: US_NVDA, US_AMD, ...
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
    列名格式: KR_005930, KR_000660, ...
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


def fetch_ashare_stocks(start_date: str = "20220101") -> dict:
    """
    逐只获取A股目标股票日线数据（通过akshare访问东方财富，国内直连）
    返回: {code: {'return': Series, 'label': Series, 'name': str}}
    """
    codes = get_all_ashare_codes()
    results = {}

    for i, (code, name) in enumerate(codes.items(), 1):
        print(f"  [{i:>2}/{len(codes)}] {name} ({code})")
        try:
            with _no_proxy():
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date, adjust="qfq"
                )

            # 兼容 akshare 不同版本列名
            col_map = {}
            for c in df.columns:
                cl = str(c)
                if "日期" in cl or cl.lower() == "date":
                    col_map[c] = "Date"
                elif "开盘" in cl or cl.lower() in ("open",):
                    col_map[c] = "Open"
                elif "收盘" in cl or cl.lower() in ("close",):
                    col_map[c] = "Close"
            df = df.rename(columns=col_map)[["Date", "Open", "Close"]].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

            intraday = (df["Close"] - df["Open"]) / df["Open"]
            results[code] = {
                "return": intraday,
                "label": (intraday > 0).astype(int),
                "name": name,
            }
        except Exception as e:
            print(f"    WARN: {name} ({code}) failed — {e}")

        # 避免 akshare 触发频率限制
        if i % 5 == 0:
            time.sleep(0.5)

    print(f"  A-share done — {len(results)}/{len(codes)} fetched OK")
    return results
