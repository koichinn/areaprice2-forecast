# -*- coding: utf-8 -*-
"""東京DAY1 予測履歴 (ドリフト補正用) を OOF キャッシュから初期化する。
出力: P:\エリアプライス2\予測\tokyo_day1_v4_pred_history.csv
 - pred_raw: ドリフト補正前の予測 (HGB+LGB ブレンド + IC ディカップリング)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"P:\エリアプライス2")
OUT = ROOT / "予測" / "tokyo_day1_v4_pred_history.csv"

df_oof = pd.read_parquet(ROOT / "学習/fast_eval_cache/oof_cache_tokyo_day1.parquet")
df_lgb = pd.read_parquet(ROOT / "学習/fast_eval_cache/oof_cache_tokyo_lgb.parquet")
df = df_oof.merge(df_lgb.drop(columns=["y", "fold_idx"], errors="ignore"), on="target_datetime")
aic = pd.read_parquet(ROOT / "学習/evaluation_system_day1_levers_da/occto_area_ic_da.parquet")
df = df.merge(aic[["target_datetime", "ic_fc_bwd_free"]], on="target_datetime", how="left").sort_values("target_datetime").reset_index(drop=True)

sys.path.insert(0, str(ROOT / "学習"))
import fast_tune_tokyo_day1 as ft

c = {k: df[k].to_numpy(float) for k in df.columns if k.startswith(("pred_", "anc_"))}
v4_hgb = ft.final_pred(ft.components(c, df["p_sys_v4"].to_numpy(float)), 0.70, 0.10)
lgbd = (0.4 * df["lgb_tok_q60"] + 0.3 * df["lgb_tok_q65"] + 0.2 * df["lgb_tok_sim_q65"] + 0.1 * df["lgb_tok_l1"]).to_numpy()
t = pd.to_datetime(df["target_datetime"])
slot = (t.dt.hour * 2 + t.dt.minute // 30).to_numpy()
is_day = ((slot >= 14) & (slot <= 41)).astype(float)
pred = (1 - 0.30 * is_day - 0.05 * (1 - is_day)) * v4_hgb + (0.30 * is_day + 0.05 * (1 - is_day)) * lgbd

tok_cal = 0.20 * c["pred_tok_cal_day"] + 0.25 * c["pred_tok_cal_wide"] + 0.25 * c["pred_tok_cal_wide_deep"] + 0.15 * c["pred_tok_cal_blk4"] + 0.15 * c["pred_tok_cal_blk6"]
tok_sim = 0.55 * c["pred_tok_sim_wide"] + 0.45 * c["pred_tok_sim_blk6"]
tok_lag1 = 0.60 * c["pred_tok_lag1_wide"] + 0.40 * c["pred_tok_lag1_blk6"]
hgb_l1 = (0.30 * tok_cal + 0.15 * tok_sim + 0.10 * tok_lag1) / 0.55
hgb_q65 = 0.60 * c["pred_tok_cal_q65"] + 0.40 * c["pred_tok_sim_q65"]
pure_direct = 0.5 * (0.70 * hgb_q65 + 0.20 * hgb_l1 + 0.10 * c["anc_cal"]) + 0.5 * lgbd
fc = df["ic_fc_bwd_free"].fillna(999.0).to_numpy(float)
pred = np.where((is_day == 1) & (fc <= 200.0), 0.40 * v4_hgb + 0.60 * pure_direct, pred)

hist = pd.DataFrame({"target_datetime": t, "pred_raw": pred, "source": "oof_seed"})
if OUT.exists():
    old = pd.read_csv(OUT, encoding="utf-8-sig", parse_dates=["target_datetime"])
    old = old[~old["target_datetime"].isin(hist["target_datetime"])]
    hist = pd.concat([hist, old]).sort_values("target_datetime")
hist.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"saved {OUT} rows={len(hist)} range={hist['target_datetime'].min()} - {hist['target_datetime'].max()}")
print(f"MAE check (vs y): {np.mean(np.abs(pred - df['y'].to_numpy(float))):.4f}")
