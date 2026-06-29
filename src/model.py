"""
model.py — 分板块预测模型
每个板块独立训练一个 RandomForest，预测当日A股板块篮子涨跌方向
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.stocks_universe import SECTORS
from src.processor import get_feature_columns


def train_sector_models(df: pd.DataFrame) -> dict:
    """
    为每个板块训练独立的分类模型。
    特征：全部美股/韩股信号（不限于同板块，因为NVDA对所有板块都有影响）
    目标：该板块等权篮子的涨跌方向
    
    Returns:
        {sector_key: {model, accuracy, prob_up, latest_date, feature_importance, backtest_7, ...}}
    """
    feature_cols = get_feature_columns(df)
    results = {}

    for sk, sd in SECTORS.items():
        label_col = f"{sk}_label"
        if label_col not in df.columns:
            continue

        subset = df[feature_cols + [label_col]].dropna()
        n = len(subset)
        if n < 60:
            print(f"  {sd['name']}: 数据不足 ({n} 行)，跳过")
            continue

        X = subset[feature_cols]
        y = subset[label_col].astype(int)

        # 时序划分：前80%训练，后20%测试
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        # 预测最新一行（今日早盘）
        latest_X = X.iloc[[-1]]
        prob = model.predict_proba(latest_X)[0]
        latest_date = X.index[-1].strftime("%Y-%m-%d")

        # 特征重要性 Top10
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]

        # 回测最近7个测试集交易日的预测结果（用于冷启动填充 PAST_7）
        backtest_7 = []
        if len(X_test) >= 1:
            test_probs = model.predict_proba(X_test)[:, 1]
            test_dates = X_test.index
            for i in range(max(0, len(X_test) - 7), len(X_test)):
                dt_str = test_dates[i].strftime("%Y-%m-%d")
                p_up = float(test_probs[i])
                pred_dir = 1 if p_up >= 0.5 else 0
                actual = int(y_test.iloc[i])
                correct = 1 if pred_dir == actual else 0
                backtest_7.append({
                    "date": dt_str,
                    "prob_up": p_up,
                    "pred_dir": pred_dir,
                    "actual_label": actual,
                    "correct": correct
                })

        results[sk] = {
            "model":           model,
            "accuracy":        acc,
            "latest_date":     latest_date,
            "prob_up":         prob[1],
            "prob_down":       prob[0],
            "top_features":    top_features,
            "sector_name":     sd["name"],
            "sector_desc":     sd["desc"],
            "n_samples":       n,
            "backtest_7":      backtest_7,
        }

    return results


def predict_individual_stocks(df: pd.DataFrame, sector_results: dict) -> dict:
    """
    用每个板块的模型对板块内个股做预测（使用同一特征集，目标换成个股标签）
    Returns: {code: {name, prob_up, sector}}
    """
    feature_cols = get_feature_columns(df)
    stock_preds = {}

    for sk, sd in SECTORS.items():
        if sk not in sector_results:
            continue
        sector_model = sector_results[sk]["model"]

        for code, name in sd["a"].items():
            label_col = f"A_{code}_{name}_Label"
            if label_col not in df.columns:
                continue

            subset = df[feature_cols + [label_col]].dropna()
            if len(subset) < 30:
                continue

            X = subset[feature_cols]
            y = subset[label_col].astype(int)
            split = int(len(subset) * 0.8)

            # 用板块级模型直接预测（不重新训练），也可选择单独训练
            latest_X = X.iloc[[-1]]
            prob = sector_model.predict_proba(latest_X)[0]

            stock_preds[code] = {
                "name":    name,
                "prob_up": prob[1],
                "sector":  sk,
                "sector_name": sd["name"],
            }

    return stock_preds


def print_report(sector_results: dict, stock_preds: dict) -> None:
    """打印每日产业链预测报告"""
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    GRAY  = "\033[90m"
    RESET = "\033[0m"

    def signal_str(prob: float) -> str:
        if prob >= 0.60:
            return f"{GREEN}↑ 偏多 {prob:.1%}{RESET} ★★★"
        elif prob >= 0.55:
            return f"{GREEN}↑ 偏多 {prob:.1%}{RESET} ★★"
        elif prob <= 0.40:
            return f"{RED}↓ 偏空 {prob:.1%}{RESET} ★★★"
        elif prob <= 0.45:
            return f"{RED}↓ 偏空 {prob:.1%}{RESET} ★★"
        else:
            return f"{GRAY}→ 中性 {prob:.1%}{RESET} ★"

    print()
    print(BOLD + "=" * 65 + RESET)
    print(BOLD + "   GPU/AI 全产业链 · 早盘预测报告" + RESET)
    print(BOLD + "=" * 65 + RESET)

    for sk, res in sector_results.items():
        print()
        print(BOLD + f"  {res['sector_name']}" + RESET)
        print(f"  {GRAY}{res['sector_desc']}{RESET}")
        print(f"  预测日期 : {res['latest_date']}")
        print(f"  板块信号 : {signal_str(res['prob_up'])}")
        print(f"  历史准确率: {res['accuracy']:.1%}  (样本量: {res['n_samples']})")
        top = "  |  ".join([f"{f}({v:.1%})" for f, v in res['top_features'][:3]])
        print(f"  关键驱动 : {top}")

        # 该板块个股
        sector_stocks = {c: p for c, p in stock_preds.items() if p["sector"] == sk}
        if sector_stocks:
            print(f"  {GRAY}── 个股预测 ──{RESET}")
            for code, p in sector_stocks.items():
                bar = "■" * int(p["prob_up"] * 10) + "□" * (10 - int(p["prob_up"] * 10))
                print(f"    {code} {p['name']:<10} [{bar}] {p['prob_up']:.1%}")

    print()
    print(BOLD + "=" * 65 + RESET)
    print(f"  {GRAY}注：概率≥55%偏多，≤45%偏空，50%附近为中性，仅供参考{RESET}")
    print(BOLD + "=" * 65 + RESET)
    print()
