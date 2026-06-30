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



import json

CACHE_FILE = "data/prediction_history.json"

def update_and_get_history(df: pd.DataFrame, sector_results: dict) -> dict:
    """
    维护一个本地的真实预测缓存，防止未来函数。
    读取历史预测，填入 df 中已知的真实涨跌结果，然后把今天的预测追加进去保存。
    冷启动时，用模型测试集回测结果填充过去 7 天。
    """
    history = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            history = json.load(f)

    # 0. 冷启动：如果缓存中已验证结果不足7天，用回测数据填充
    verified_dates = [d for d in history.keys()
                      if any(v.get("actual_label") is not None for v in history[d].values())]
    if len(verified_dates) < 7:
        # 从模型的 backtest_7 中提取回测结果写入缓存
        for sk, res in sector_results.items():
            for bt in res.get("backtest_7", []):
                dt = bt["date"]
                if dt not in history:
                    history[dt] = {}
                if sk not in history[dt]:
                    history[dt][sk] = {
                        "prob_up": bt["prob_up"],
                        "pred_dir": bt["pred_dir"],
                        "actual_label": bt["actual_label"],
                        "correct": bt["correct"]
                    }

    # 1. 更新过去预测的真实结果
    for date_str, daily_data in history.items():
        if date_str in df.index.strftime("%Y-%m-%d"):
            idx = df.index[df.index.strftime("%Y-%m-%d") == date_str][0]
            for sk, s_data in daily_data.items():
                if s_data.get("correct") is not None:
                    continue  # 已有结果
                label_col = f"{sk}_label"
                if label_col in df.columns and not pd.isna(df.loc[idx, label_col]):
                    actual = int(df.loc[idx, label_col])
                    s_data["actual_label"] = actual
                    if s_data.get("pred_dir") is not None:
                        s_data["correct"] = 1 if s_data["pred_dir"] == actual else 0

    # 2. 追加今天的预测
    today_data = {}
    for sk, res in sector_results.items():
        prob = float(res["prob_up"])
        th_up = float(res.get("threshold_up", 0.55))
        th_down = float(res.get("threshold_down", 0.45))
        # 用每个板块的多空不对称最优阈值判断
        if th_down < prob < th_up:
            pred_dir = None
        else:
            pred_dir = 1 if prob >= th_up else 0
        today_data[sk] = {
            "prob_up": prob,
            "pred_dir": pred_dir,
            "actual_label": None,
            "correct": None
        }
    history[TODAY_STR] = today_data

    with open(CACHE_FILE, "w") as f:
        json.dump(history, f, indent=4)

    return history


def generate_markdown(sector_results: dict, history: dict, df: pd.DataFrame) -> str:
    lines = []
    
    lines.append(f"## 🤖 AI-Quant System")
    lines.append(f"> **Date**: {TODAY_STR} | **Target**: A-Share")
    lines.append("")
    
    # 筛选出已经有真实结果的历史日期，取最近7天
    past_dates = sorted([d for d in history.keys() if d != TODAY_STR and any(v.get("actual_label") is not None for v in history[d].values())])
    last_7_dates = past_dates[-7:]
    
    for i, (sk, res) in enumerate(sector_results.items(), 1):
        sector_name = res["sector_name"]
        prob = res["prob_up"]
        th_up = res.get("threshold_up", 0.55)
        th_down = res.get("threshold_down", 0.45)
        winrate = res.get("filtered_winrate", 0)
        
        results = []
        for d in last_7_dates:
            d_data = history[d].get(sk)
            if d_data and d_data.get("actual_label") is not None:
                if d_data.get("pred_dir") is None:
                    results.append("➖")  # 中性
                else:
                    results.append("✅" if d_data.get("correct") == 1 else "❌")
            else:
                results.append("⚪")  # 无数据
        hist_str = "".join(results)
            
        if prob >= th_up:
            prob_color = "info"
            dir_icon = "UP"
        elif prob <= th_down:
            prob_color = "warning"
            dir_icon = "DOWN"
        else:
            prob_color = "comment"
            dir_icon = "NEUTRAL"
        
        lines.append(f"**[{i:02d}] {sector_name}**")
        lines.append(f"> <font color=\"{prob_color}\">PROB_UP</font>: **{prob:.1%}** [{dir_icon}]")
        lines.append(f"> <font color=\"comment\">PAST_7 </font>: [{hist_str}] | WR: {winrate:.0%} (up={th_up:.2f}/dn={th_down:.2f})")
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  GPU/AI 产业链报告生成器")
    print("=" * 60)

    print("\n[1/4] 获取美股数据...")
    us_returns = fetch_us_stocks("2025-09-01")
    print("US data fetched.")

    kr_gaps = fetch_korea_stocks("2025-09-01")
    print("KR data fetched.")

    ashare_data = fetch_ashare_stocks("2025-09-01")

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

    # ==========================================
    # 【Bugfix】: 生成“今天”的最新特征向量
    # 因为今天A股还没收盘，df 里的最后一行其实是“昨天”的训练数据。
    # 真正的预测必须用昨晚的美股收盘和今天早上的韩股开盘。
    # ==========================================
    today_date = pd.to_datetime(datetime.now().date())
    
    # 1. 提取昨晚美股最新收盘
    latest_us = us_returns.iloc[-1:] 
    
    # 2. 提取今天早上韩国10:00截面
    latest_kr = kr_gaps.iloc[-1:]
    
    # 3. 构造 today_features
    today_features = {}
    for col in latest_us.columns:
        today_features[col] = latest_us[col].values[0]
    for col in latest_kr.columns:
        today_features[col] = latest_kr[col].values[0]

    for sk, sd in SECTORS.items():
        us_cols = [f"US_{t}" for t in sd["us"] if f"US_{t}" in latest_us.columns]
        if us_cols:
            today_features[f"{sk}_US"] = latest_us[us_cols].mean(axis=1).values[0]
            
        kr_cols = [f"KR_{t.split('.')[0]}" for t in sd["kr"] if f"KR_{t.split('.')[0]}" in latest_kr.columns]
        if kr_cols:
            today_features[f"{sk}_KR"] = latest_kr[kr_cols].mean(axis=1).values[0]
            
    today_df = pd.DataFrame([today_features], index=[today_date])
    
    # 将 today_df 送入训练好的模型中进行独立预测，覆盖原本 sector_results 里默认提取的历史最后一天结果
    from src.processor import get_feature_columns
    feature_cols = get_feature_columns(df)
    for sk, res in sector_results.items():
        clf = res["model"]
        X_today = today_df[[c for c in feature_cols if sk in c or not any(s in c for s in SECTORS.keys() if s != sk)]]
        
        # 为了兼容特征列，补齐可能缺失的列
        for col in clf.feature_names_in_:
            if col not in X_today.columns:
                X_today[col] = latest_us[col].values[0] if col in latest_us.columns else 0
                
        prob = clf.predict_proba(X_today[clf.feature_names_in_])[0][1]
        res["prob_up"] = prob

    print("\n生成/更新真实预测缓存...")
    history = update_and_get_history(df, sector_results)

    print("\n生成报告...")
    md_content = generate_markdown(sector_results, history, df)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n✅ 报告已生成: {REPORT_FILE}")
    print(f"   板块数: {len(sector_results)}, 回测天数: {BACKTEST_DAYS}")
    
    # 微信推送
    from src.notifier import send_wechat_webhook
    send_wechat_webhook(md_content)


if __name__ == "__main__":
    main()
