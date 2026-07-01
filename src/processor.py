"""
processor.py — 多市场数据对齐与特征工程
- 美股因子：T-1对齐（merge_asof backward, 严格不含当日）
- 韩股因子：T日对齐（直接reindex）
- A股目标：等权板块篮子 + 个股标签
"""
from __future__ import annotations


import pandas as pd
import numpy as np
from src.stocks_universe import SECTORS
from src.advanced_features import add_advanced_features


def align_us_to_ashare(us_returns: pd.DataFrame, ashare_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    将美股收益率对齐到A股交易日，采用严格T-1逻辑：
    对每个A股交易日T，取最近一个美股T-1收盘（不含T日本身）
    """
    base = pd.DataFrame({"Date": ashare_dates}).sort_values("Date")
    us_reset = us_returns.reset_index().sort_values("Date")
    us_reset = us_reset.rename(columns={"index": "Date"}) if "index" in us_reset.columns else us_reset

    aligned = pd.merge_asof(
        base,
        us_reset,
        on="Date",
        direction="backward",
        allow_exact_matches=False,   # 严格T-1，不取当日
    )
    aligned = aligned.set_index("Date")
    return aligned


def build_sector_features(
    us_aligned: pd.DataFrame,
    kr_gaps: pd.DataFrame,
    ashare_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    为每个板块生成聚合信号（板块内股票等权平均）
    - {sector}_US: 美股板块信号（T-1）
    - {sector}_KR: 韩股板块信号（T日跳空）
    """
    kr_aligned = kr_gaps.reindex(ashare_dates)
    sector_feats: dict[str, pd.Series] = {}

    for sk, sd in SECTORS.items():
        us_cols = [f"US_{t}" for t in sd["us"] if f"US_{t}" in us_aligned.columns]
        if us_cols:
            sector_feats[f"{sk}_US"] = us_aligned[us_cols].mean(axis=1)

        kr_cols = [f"KR_{t.split('.')[0]}" for t in sd["kr"]
                   if f"KR_{t.split('.')[0]}" in kr_aligned.columns]
        if kr_cols:
            sector_feats[f"{sk}_KR"] = kr_aligned[kr_cols].mean(axis=1)

    return pd.DataFrame(sector_feats, index=ashare_dates)


def build_ashare_baskets(
    ashare_data: dict,
    ashare_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建两类目标：
    1. sector_df: 每板块等权篮子的收益率 & 方向标签
    2. stock_df:  每只个股的方向标签
    """
    sector_rows: dict[str, pd.Series] = {}
    stock_rows:  dict[str, pd.Series] = {}

    for sk, sd in SECTORS.items():
        returns_list = []
        for code, name in sd["a"].items():
            if code in ashare_data:
                r = ashare_data[code]["return"].reindex(ashare_dates)
                returns_list.append(r)
                stock_rows[f"A_{code}_{name}_Label"] = ashare_data[code]["label"].reindex(ashare_dates)

        if returns_list:
            basket = pd.concat(returns_list, axis=1).mean(axis=1)
            sector_rows[f"{sk}_basket_ret"] = basket
            sector_rows[f"{sk}_label"] = basket

    return (
        pd.DataFrame(sector_rows, index=ashare_dates),
        pd.DataFrame(stock_rows, index=ashare_dates),
    )


def build_full_dataset(
    us_aligned: pd.DataFrame,
    kr_gaps: pd.DataFrame,
    sector_features: pd.DataFrame,
    sector_targets: pd.DataFrame,
    stock_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    合并所有数据，以板块标签列不全为NaN为基准删除缺失行
    """
    kr_aligned = kr_gaps.reindex(sector_features.index)

    all_data = pd.concat([
        us_aligned,
        kr_aligned,
        sector_features,
        sector_targets,
        stock_labels,
    ], axis=1)

    # 只要有任一板块标签存在就保留该行，但强制保留最后一行（今日）用于预测
    label_cols = [c for c in all_data.columns if c.endswith("_label")]
    mask = all_data[label_cols].notna().any(axis=1)
    if len(mask) > 0:
        mask.iloc[-1] = True
    all_data = all_data[mask]
    
    # 注入高级特征
    all_data = add_advanced_features(all_data, us_aligned, kr_aligned)

    print(f"  Full dataset: {len(all_data)} rows × {len(all_data.columns)} cols")
    return all_data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """提取所有特征列（US_ / KR_ / _US / _KR 开头结尾的列）"""
    return [
        c for c in df.columns
        if c.startswith("US_") or c.startswith("KR_") or c.startswith("ADV_")
           or c.endswith("_US") or c.endswith("_KR")
    ]
