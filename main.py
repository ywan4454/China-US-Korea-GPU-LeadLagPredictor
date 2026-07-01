"""
main.py — GPU/AI 产业链 Lead-Lag 预测系统入口
运行方式:
  本地（需代理访问Yahoo）: HTTP_PROXY=http://127.0.0.1:1087 NO_PROXY=eastmoney.com python main.py
  GitHub Actions（无需代理）: python main.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import fetch_us_stocks, fetch_korea_stocks, fetch_ashare_stocks
from src.processor import (
    align_us_to_ashare,
    build_sector_features,
    build_ashare_baskets,
    build_full_dataset,
)
from src.model import train_sector_models, predict_individual_stocks, print_report


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 65)
    print("  GPU/AI 全产业链 Lead-Lag 预测系统  (5大板块 · 22只A股)")
    print("=" * 65)

    # ── 1. 获取数据 ──────────────────────────────────────────────
    print("\n[Step 1/4] 获取美股因子数据...")
    us_returns = fetch_us_stocks("2025-09-01")
    print("US data fetched.")

    kr_gaps = fetch_korea_stocks("2025-09-01")
    print("KR data fetched.")

    print("\n[Step 3/4] 获取A股目标数据（22只个股）...")
    ashare_data = fetch_ashare_stocks("2025-09-01")

    if not ashare_data:
        print("ERROR: 无法获取任何A股数据，请检查网络连接。")
        sys.exit(1)

    # ── 2. 对齐与特征工程 ────────────────────────────────────────
    print("\n[Step 4/4] 数据对齐与特征工程...")

    # 以所有A股数据的日期并集为基准
    all_dates = None
    for d in ashare_data.values():
        idx = d["return"].dropna().index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    
    import pandas as pd
    from datetime import datetime
    today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
    if today not in all_dates:
        all_dates = all_dates.append(pd.DatetimeIndex([today]))
        
    all_dates = all_dates.sort_values()

    us_aligned      = align_us_to_ashare(us_returns, all_dates)
    sector_features = build_sector_features(us_aligned, kr_gaps, all_dates)
    sector_targets, stock_labels = build_ashare_baskets(ashare_data, all_dates)
    df = build_full_dataset(us_aligned, kr_gaps, sector_features, sector_targets, stock_labels)

    df.to_csv("data/full_dataset.csv")
    print(f"  Dataset saved → data/full_dataset.csv")

    # ── 3. 训练与预测 ────────────────────────────────────────────
    print("\n训练各板块模型...")
    sector_results = train_sector_models(df)

    print("\n预测个股...")
    stock_preds = predict_individual_stocks(df, sector_results)

    # ── 4. 输出报告 ──────────────────────────────────────────────
    print_report(sector_results, stock_preds, df)


if __name__ == "__main__":
    main()
