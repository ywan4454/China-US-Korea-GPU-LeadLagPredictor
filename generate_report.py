"""
generate_report.py — 生成每日预测报告 + 过去7天回测准确率
输出: data/daily_report.md
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

from src.data_fetcher import fetch_us_stocks, fetch_korea_stocks, fetch_ashare_stocks
from src.processor import (
    align_us_to_ashare, build_sector_features,
    build_ashare_baskets, build_full_dataset,
)
from src.model import train_sector_models
from src.stocks_universe import SECTORS

# ── 配置 ──────────────────────────────────────────────────────────
BACKTEST_DAYS = 7        # 回测最近N个A股交易日
REPORT_FILE   = "data/daily_report.md"
TODAY_STR     = datetime.now().strftime("%Y-%m-%d")



def build_backtest_table(df: pd.DataFrame, sector_results: dict) -> dict:
    """
    对过去 BACKTEST_DAYS 个交易日，用已训练好的模型做预测，与实际对比
    返回 {sector_key: DataFrame with columns [date, prob_up, actual_label, correct]}
    """
    from src.processor import get_feature_columns
    feature_cols = get_feature_columns(df)
    backtest = {}

    for sk, res in sector_results.items():
        label_col = f"{sk}_label"
        if label_col not in df.columns:
            continue

        subset = df[feature_cols + [label_col]].dropna()
        if len(subset) < BACKTEST_DAYS + 10:
            continue

        # 取最后 BACKTEST_DAYS 行作为回测窗口
        bt_window = subset.iloc[-(BACKTEST_DAYS):]
        X_bt = bt_window[feature_cols]
        y_bt = bt_window[label_col].astype(int)

        probs = res["model"].predict_proba(X_bt)[:, 1]
        preds = (probs >= 0.5).astype(int)
        correct = (preds == y_bt.values).astype(int)

        backtest[sk] = pd.DataFrame({
            "date":         bt_window.index.strftime("%Y-%m-%d"),
            "prob_up":      probs,
            "pred_dir":     ["↑" if p == 1 else "↓" for p in preds],
            "actual_label": y_bt.values,
            "actual_dir":   ["↑" if a == 1 else "↓" for a in y_bt.values],
            "correct":      correct,
        })

    return backtest


def generate_markdown(sector_results: dict, backtest: dict, df: pd.DataFrame) -> str:
    lines = []
    
    lines.append(f"## 🤖 AI-Quant System")
    lines.append(f"> **Date**: {TODAY_STR} | **Target**: A-Share")
    lines.append("")
    
    for i, (sk, res) in enumerate(sector_results.items(), 1):
        sector_name = res["sector_name"]
        prob = res["prob_up"]
        
        hist_str = ""
        if sk in backtest:
            bt_df = backtest[sk]
            results = []
            for _, row in bt_df.iterrows():
                results.append("✅" if row["correct"] == 1 else "❌")
            hist_str = "".join(results)
            
        prob_color = "info" if prob >= 0.5 else "warning"
        dir_icon = "📈" if prob >= 0.5 else "📉"
        
        lines.append(f"**[{i:02d}] {sector_name}**")
        lines.append(f"> <font color=\"{prob_color}\">PROB_UP</font>: **{prob:.1%}** {dir_icon}")
        lines.append(f"> <font color=\"comment\">PAST_7 </font>: [{hist_str}]")
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  GPU/AI 产业链报告生成器")
    print("=" * 60)

    print("\n[1/4] 获取美股数据...")
    us_returns = fetch_us_stocks("2022-01-01")

    print("\n[2/4] 获取韩股数据...")
    kr_gaps = fetch_korea_stocks("2022-01-01")

    print("\n[3/4] 获取A股数据...")
    ashare_data = fetch_ashare_stocks("2022-01-01")

    print("\n[4/4] 对齐 & 训练模型...")
    all_dates = None
    for d in ashare_data.values():
        idx = d["return"].dropna().index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    all_dates = all_dates.sort_values()

    us_aligned      = align_us_to_ashare(us_returns, all_dates)
    sector_features = build_sector_features(us_aligned, kr_gaps, all_dates)
    sector_targets, stock_labels = build_ashare_baskets(ashare_data, all_dates)
    df = build_full_dataset(us_aligned, kr_gaps, sector_features, sector_targets, stock_labels)
    df.to_csv("data/full_dataset.csv")

    print("\n训练模型...")
    sector_results = train_sector_models(df)

    print("\n生成回测数据...")
    backtest = build_backtest_table(df, sector_results)

    print("\n生成报告...")
    md_content = generate_markdown(sector_results, backtest, df)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ 报告已生成: {REPORT_FILE}")
    print(f"   板块数: {len(sector_results)}, 回测天数: {BACKTEST_DAYS}")
    
    # 微信推送
    from src.notifier import send_wechat_webhook
    send_wechat_webhook(md_content)


if __name__ == "__main__":
    main()
