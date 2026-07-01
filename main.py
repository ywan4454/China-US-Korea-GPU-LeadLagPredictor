import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import fetch_us_stocks, fetch_korea_stocks, fetch_ashare_stocks
from src.processor import (
    align_us_to_ashare, build_sector_features,
    build_ashare_baskets, build_full_dataset
)
from src.model import train_sector_models, print_report, get_3class_label
from src.stocks_universe import SECTORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def build_dataset_pipeline():
    print("\n[1/4] 获取美股数据...")
    us_returns = fetch_us_stocks("2025-09-01")
    print("US data fetched.")

    kr_gaps = fetch_korea_stocks("2025-09-01")
    print("KR data fetched.")

    print("\n[3/4] 获取A股数据...")
    ashare_data = fetch_ashare_stocks("2025-09-01")
    if not ashare_data:
        print("ERROR: 无法获取任何A股数据，请检查网络连接。")
        sys.exit(1)

    print("\n[4/4] 数据对齐与特征工程...")
    all_dates = None
    for d in ashare_data.values():
        idx = d["return"].dropna().index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    
    today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
    if today not in all_dates:
        all_dates = all_dates.append(pd.DatetimeIndex([today]))
        
    all_dates = all_dates.sort_values()

    us_aligned      = align_us_to_ashare(us_returns, all_dates)
    sector_features = build_sector_features(us_aligned, kr_gaps, all_dates)
    sector_targets, stock_labels = build_ashare_baskets(ashare_data, all_dates)
    df = build_full_dataset(us_aligned, kr_gaps, sector_features, sector_targets, stock_labels)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/full_dataset.csv")
    print(f"  Dataset saved → data/full_dataset.csv")
    return df

def run_prediction(args):
    print("=" * 65)
    print("  GPU/AI 产业链日常预测模式")
    print("=" * 65)
    df = build_dataset_pipeline()
    print("\n训练模型...")
    sector_results = train_sector_models(df)
    print_report(sector_results, df)

def run_backtest(args):
    print("=" * 85)
    print(f"  过去 {args.days} 个交易日回测 (特征隔离 + 重力惩罚版)")
    print("=" * 85)
    
    if not os.path.exists("data/full_dataset.csv"):
        print("未找到数据集，正在拉取数据...")
        df = build_dataset_pipeline()
    else:
        df = pd.read_csv("data/full_dataset.csv", index_col=0, parse_dates=True)
        
    label_cols = [c for c in df.columns if c.endswith("_label")]
    mask = df[label_cols].notna().any(axis=1)
    valid_df = df[mask].copy()

    last_n_dates = valid_df.index[-args.days:]
    results = []

    for sk, sd in SECTORS.items():
        label_col = f"{sk}_label"
        if label_col not in valid_df.columns: continue
            
        sector_us_cols = [f"US_{t}" for t in sd["us"].keys()]
        sector_kr_cols = [f"KR_{t.split('.')[0]}" for t in sd["kr"].keys()]
        agg_cols = [f"{sk}_US", f"{sk}_KR"]
        adv_cols = [c for c in df.columns if c.startswith(f"ADV_{sk}_") or c.startswith("ADV_Flag_")]
        
        my_cols = sector_us_cols + sector_kr_cols + agg_cols + adv_cols
        sector_feature_cols = [c for c in my_cols if c in df.columns]
        
        subset = valid_df[sector_feature_cols + [label_col]].dropna()
        if len(subset) < 60: continue
            
        X = subset[sector_feature_cols]
        y = subset[label_col].apply(get_3class_label)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            class_weight={1: 5.0, -1: 5.0, 0: 1.0}, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        importance = dict(zip(sector_feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:3]
        top_feature_names = [f[0] for f in top_features]
        
        common_dates = last_n_dates.intersection(subset.index)
        X_last_n = subset.loc[common_dates, sector_feature_cols]
        y_last_n = subset.loc[common_dates, label_col]
        
        preds_class = model.predict(X_last_n)
        
        for date, actual_ret, pred_class in zip(common_dates, y_last_n, preds_class):
            reason_parts = []
            for tf in top_feature_names:
                val = X_last_n.loc[date, tf]
                reason_parts.append(f"{tf.replace('US_', '').replace('KR_', '')}: {val:+.1%}")
                
            results.append({
                "date": date, "sector": sd["name"], "actual_ret": actual_ret,
                "pred_class": int(pred_class), "reason": ", ".join(reason_parts)
            })

    res_df = pd.DataFrame(results)
    days_map = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
    
    if len(res_df) == 0:
        print("无可用的回测结果。")
        return

    for d in last_n_dates:
        day_str = f"{d.strftime('%Y-%m-%d')} (周{days_map[d.weekday()]})"
        print(f"\n📅 {day_str}")
        day_data = res_df[res_df["date"] == d]
        
        for _, row in day_data.iterrows():
            actual_ret = row["actual_ret"]
            pred_class = row["pred_class"]
            act_class = get_3class_label(actual_ret)
            
            act_dir = "↑多" if act_class == 1 else ("↓空" if act_class == -1 else "→平")
            pred_dir = "↑多" if pred_class == 1 else ("↓空" if pred_class == -1 else "→平")
            
            mark = "✅命中" if act_class == pred_class else ("➖踏空" if pred_class == 0 else ("⚠️误报" if act_class == 0 else "❌反向"))
            sec = row['sector'].replace('板块1·', '').replace('板块2·', '').replace('板块3·', '').replace('板块4·', '').replace('板块5·', '')
            print(f"  {sec:5} | 预测: {pred_dir} | 实际: {actual_ret:>6.2%} ({act_dir}) | {mark:4} | 专属动因: {row['reason']}")
    print("\n" + "="*85 + "\n")

def run_eval(args):
    print("==== 各板块全部Sample Winrate (Train + Test) ====")
    if not os.path.exists("data/full_dataset.csv"):
        print("未找到数据集，正在拉取数据...")
        df = build_dataset_pipeline()
    else:
        df = pd.read_csv("data/full_dataset.csv", index_col=0, parse_dates=True)
        
    for sk, sd in SECTORS.items():
        label_col = f"{sk}_label"
        if label_col not in df.columns: continue
        sector_us_cols = [f"US_{t}" for t in sd["us"].keys()]
        sector_kr_cols = [f"KR_{t.split('.')[0]}" for t in sd["kr"].keys()]
        agg_cols = [f"{sk}_US", f"{sk}_KR"]
        adv_cols = [c for c in df.columns if c.startswith(f"ADV_{sk}_") or c.startswith("ADV_Flag_")]
        
        my_cols = sector_us_cols + sector_kr_cols + agg_cols + adv_cols
        sector_feature_cols = [c for c in my_cols if c in df.columns]

        subset = df[sector_feature_cols + [label_col]].dropna()
        n = len(subset)
        if n < 60: continue

        X = subset[sector_feature_cols]
        y = subset[label_col].apply(get_3class_label)
        
        split = int(n * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        
        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight={1: 5.0, -1: 5.0, 0: 1.0}, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        acc_all = accuracy_score(y, model.predict(X))
        print(f"{sd['name']:<15} | 全部样本数: {n} | 全样本胜率: {acc_all:.2%}")
    print("==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU/AI 全产业链 Lead-Lag 预测系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    parser_run = subparsers.add_parser("run", help="运行日常早盘预测 (默认)")
    
    parser_backtest = subparsers.add_parser("backtest", help="运行回测")
    parser_backtest.add_argument("--days", type=int, default=7, help="回测天数")
    
    parser_eval = subparsers.add_parser("eval", help="评估全样本胜率")
    
    args = parser.parse_args()
    
    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        run_prediction(args)
