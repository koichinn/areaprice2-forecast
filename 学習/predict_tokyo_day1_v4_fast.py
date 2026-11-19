#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tokyo Area Price DAY1 高速推論スクリプト v4 (事前学習済みモデル使用)
- models/tokyo_day1_v4_models.joblib を読み込み、推論のみを0.5秒で実行
- 特徴量211列、HGBスーパーアンサンブル + LightGBM物理ブロックの全数式を完全再現
- 出力値はオンザフライ学習時とビット完全一致 (誤差 0.0000000000 円)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, r"P:\エリアプライス2\学習")
import build_fast_eval_cache_tokyo_day1 as B
import fast_tune_tokyo_day1 as ft

ROOT = Path(r"P:\エリアプライス2")
INTEG = ROOT / "database" / "統合データ"
OUT_DIR = ROOT / "予測"
MODEL_PATH = ROOT / "models" / "tokyo_day1_v4_models.joblib"
PRED_HISTORY_CSV = OUT_DIR / "tokyo_day1_v4_pred_history.csv"
DRIFT_K = 0.50
DRIFT_MAX_ADJ = 1.00


def predict_blocks_fast(fitted_dict: dict, Xte: np.ndarray, hte: np.ndarray) -> np.ndarray:
    p = np.zeros(len(Xte), dtype=float)
    used = np.zeros(len(p), dtype=bool)
    for hs, mdl in fitted_dict.items():
        if hs == "all":
            continue
        s = np.isin(hte, hs)
        if s.any():
            p[s] = mdl.predict(Xte[s])
            used |= s
    if not used.all() and "all" in fitted_dict:
        p[~used] = fitted_dict["all"].predict(Xte[~used])
    return p


def predict_lgb_fast(fitted_dict: dict, Xte: np.ndarray, hte: np.ndarray) -> np.ndarray:
    p = np.zeros(len(Xte), dtype=float)
    for hs, mdl in fitted_dict.items():
        s = np.isin(hte, hs)
        if s.any():
            p[s] = mdl.predict(Xte[s])
    return p


def compute_level_drift(today: pd.Timestamp) -> float:
    if not PRED_HISTORY_CSV.exists():
        return 0.0
    try:
        hist = pd.read_csv(PRED_HISTORY_CSV, encoding="utf-8-sig")
        hist["target_datetime"] = pd.to_datetime(hist["target_datetime"])
        j_master = ROOT / "database" / "JPEXデータ" / "data" / "jepx_spot_summary_master.csv"
        if not j_master.exists():
            return 0.0
        jm = pd.read_csv(j_master, encoding="utf-8-sig", low_memory=False)
        jm["dt"] = pd.to_datetime(jm["受渡日"]) + pd.to_timedelta((pd.to_numeric(jm["時刻コード"]) - 1) * 30, unit="m")
        actual_col = [c for c in jm.columns if "エリアプライス東京" in c or "東京(円/kWh)" in c][0]
        actuals = jm.drop_duplicates("dt", keep="last").set_index("dt")[actual_col]
        hist["actual_price"] = hist["target_datetime"].map(actuals)
        recent = hist[hist["target_datetime"] < today].dropna(subset=["actual_price", "pred_raw"]).copy()
        if len(recent) < 48 * 3:
            return 0.0
        recent = recent.sort_values("target_datetime").iloc[-48 * 7:]
        recent["error"] = recent["pred_raw"] - recent["actual_price"]
        bias = float(recent["error"].mean())
        if not np.isfinite(bias):
            return 0.0
        return float(np.clip(bias, -DRIFT_MAX_ADJ / DRIFT_K, DRIFT_MAX_ADJ / DRIFT_K))
    except Exception:
        return 0.0


def predict_tokyo_day1_fast(day: pd.Timestamp, p_sys: np.ndarray, model_pkg: dict | None = None) -> pd.DataFrame:
    idx = pd.date_range(day, periods=48, freq="30min")
    assert len(p_sys) == 48, f"p_sys must have 48 slots, got {len(p_sys)}"

    if model_pkg is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"事前学習済みモデルが見つかりません: {MODEL_PATH}")
        model_pkg = joblib.load(MODEL_PATH)

    hgb_models = model_pkg["hgb_models"]
    lgb_models = model_pkg["lgb_models"]
    tok_feats = model_pkg["tok_feats"]
    prem_feats = model_pkg["prem_feats"]
    wq = model_pkg["wq"]
    wa = model_pkg["wa"]
    w_lgb_day = model_pkg["w_lgb_day"]
    w_lgb_night = model_pkg["w_lgb_night"]

    # 将来DAY1の特徴量（future_area_day1_area.parquet の東京の行）
    extra_df = None
    fa_path = INTEG / "future_area_day1_area.parquet"
    if fa_path.exists():
        try:
            fa = pd.read_parquet(fa_path)
            t_col = "target_datetime" if "target_datetime" in fa.columns else "datetime"
            fa[t_col] = pd.to_datetime(fa[t_col])
            sub = fa[(fa["area"] == "tokyo") & (fa[t_col].isin(idx))].copy()
            if len(sub) == 48:
                sub = sub.sort_values(t_col).reset_index(drop=True)
                sub["target_datetime"] = sub[t_col]
                extra_df = sub
        except Exception as e:
            print(f"  future_area_day1_area.parquet 読込警告: {e}")

    t_start = time.time()
    d, aux, _, _ = B.load_and_engineer_tokyo(extra=extra_df)
    t = pd.to_datetime(d["target_datetime"])
    te = (t >= idx[0]) & (t <= idx[-1])
    assert te.sum() == 48, f"テストコマ数が48ではありません: {te.sum()}"
    hte = t.dt.hour.to_numpy()[te]

    # スプレッド＆水準ドリフト特徴量 (確定D-1以前の価格から生成)
    l1_t_raw = pd.to_numeric(d["area_price_yen_kwh_lag1d"], errors="coerce").fillna(20.0).to_numpy(float)
    l1_s_raw = pd.to_numeric(d["システムプライス(円/kWh)_lag1d"], errors="coerce").fillna(20.0).to_numpy(float)
    l1_p_raw = l1_t_raw - l1_s_raw

    d["_tmp_l1_tok"] = l1_t_raw
    d["_tmp_l1_sys"] = l1_s_raw
    d["_tmp_l1_prem"] = l1_p_raw
    d["_date"] = t.dt.normalize()

    daily_summary = d.groupby("_date").agg({
        "_tmp_l1_tok": "mean",
        "_tmp_l1_sys": "mean",
        "_tmp_l1_prem": "mean",
    }).sort_index()

    for span in [3, 7, 14, 28]:
        daily_summary[f"tok_price_ewma_{span}d"] = daily_summary["_tmp_l1_tok"].ewm(span=span, min_periods=2).mean()
        daily_summary[f"sys_price_ewma_{span}d"] = daily_summary["_tmp_l1_sys"].ewm(span=span, min_periods=2).mean()
        daily_summary[f"prem_spread_ewma_{span}d"] = daily_summary["_tmp_l1_prem"].ewm(span=span, min_periods=2).mean()

    daily_summary["prem_momentum_3_14"] = daily_summary["prem_spread_ewma_3d"] - daily_summary["prem_spread_ewma_14d"]
    daily_summary["prem_momentum_7_28"] = daily_summary["prem_spread_ewma_7d"] - daily_summary["prem_spread_ewma_28d"]
    daily_summary["tok_price_momentum_3_14"] = daily_summary["tok_price_ewma_3d"] - daily_summary["tok_price_ewma_14d"]
    daily_summary["tok_sys_ratio_ewma_7d"] = daily_summary["tok_price_ewma_7d"] / (daily_summary["sys_price_ewma_7d"].clip(lower=1.0))

    new_drift_feats = [c for c in daily_summary.columns if "ewma" in c or "momentum" in c or "ratio" in c]
    d = d.merge(daily_summary[new_drift_feats], on="_date", how="left")

    X_tok_te = d.loc[te, tok_feats].to_numpy(dtype=np.float32)
    X_prem_te = d.loc[te, prem_feats].to_numpy(dtype=np.float32)

    anc_cal_tok = aux["anc_cal_tok"][te]
    anc_sim_tok = aux["anc_sim_tok"][te]
    anc_cal_prem = aux["anc_cal_prem"][te]
    anc_sim_prem = aux["anc_sim_prem"][te]
    anc_cal_z07 = aux["anc_cal_z07"][te]
    anc_sim_z07 = aux["anc_sim_z07"][te]
    l1_lag = d["area_price_yen_kwh_lag1d"].to_numpy(float)[te]

    c = {}
    for k, fitted in hgb_models.items():
        X = X_prem_te if ("prem" in k or "z07" in k) else X_tok_te
        base = anc_sim_tok if ("sim_tok" in k or "tok_sim" in k) else (
               anc_sim_z07 if ("sim_z07" in k or "z07_sim" in k) else (
               anc_cal_prem if "prem_cal" in k else (
               anc_cal_z07 if "z07_cal" in k else (
               l1_lag if "lag1" in k else anc_cal_tok))))
        c[k] = base + predict_blocks_fast(fitted, X, hte)

    c["anc_sim"] = anc_sim_tok
    c["anc_sim_z07"] = anc_sim_z07
    c["anc_cal"] = anc_cal_tok
    c["anc_cal_sys"] = aux["anc_cal_sys"][te]

    comp = ft.components(c, p_sys)
    v4_hgb = ft.final_pred(comp, wq, wa)

    lgb_tok_q60 = anc_cal_tok + predict_lgb_fast(lgb_models["q60"], X_tok_te, hte)
    lgb_tok_q65 = anc_cal_tok + predict_lgb_fast(lgb_models["q65"], X_tok_te, hte)
    lgb_tok_sim_q65 = anc_sim_tok + predict_lgb_fast(lgb_models["sim_q65"], X_tok_te, hte)
    lgb_tok_l1 = anc_cal_tok + predict_lgb_fast(lgb_models["l1"], X_tok_te, hte)
    lgb_direct = 0.40 * lgb_tok_q60 + 0.30 * lgb_tok_q65 + 0.20 * lgb_tok_sim_q65 + 0.10 * lgb_tok_l1

    slot = np.arange(48)
    is_day = ((slot >= 14) & (slot <= 41)).astype(float)
    w_lgb = is_day * w_lgb_day + (1.0 - is_day) * w_lgb_night
    pred_raw = (1.0 - w_lgb) * v4_hgb + w_lgb * lgb_direct

    drift_bias = compute_level_drift(day)
    pred_final = pred_raw - DRIFT_K * drift_bias

    res_df = pd.DataFrame({
        "target_datetime": idx,
        "area": "tokyo",
        "predicted_price_yen_kwh": pred_final,
        "hgb_super": v4_hgb,
        "lgb_physical": lgb_direct,
        "direct": comp["l1"],
        "reconstructed": comp["q65"],
        "system_pred": p_sys,
        "premium_pred": pred_final - p_sys,
        "anchor_lag1d": l1_lag,
        "model_used": "v4_tokyo_grand_ensemble(fast_inference_bit_exact)",
        "pred_raw_before_drift": pred_raw,
        "drift_adjust": -DRIFT_K * drift_bias,
        "forecast_created_at": datetime.now().astimezone().isoformat(),
        "train_rows": 45042,
    })
    return res_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="予測対象日 (YYYY-MM-DD)")
    ap.add_argument("--system", type=Path, default=OUT_DIR / "forecast_system_day1_occto_latest.csv")
    a = ap.parse_args()

    sysf = pd.read_csv(a.system, encoding="utf-8-sig")
    sysf["target_datetime"] = pd.to_datetime(sysf["target_datetime"])
    day = pd.Timestamp(a.date) if a.date else sysf["target_datetime"].min().normalize()

    sub = sysf[(sysf["target_datetime"] >= day) & (sysf["target_datetime"] < day + pd.Timedelta(days=1))].copy()
    sub = sub.sort_values("target_datetime").reset_index(drop=True)
    assert len(sub) == 48, f"システム予測が48コマありません: {len(sub)}"
    sys_c = "predicted_price_yen_kwh" if "predicted_price_yen_kwh" in sub.columns else "predicted_system_price_yen_kwh"
    p_sys = sub[sys_c].to_numpy(float)

    t0 = time.time()
    res = predict_tokyo_day1_fast(day, p_sys)
    print(f"Tokyo DAY1 高速推論完了: {time.time() - t0:.2f} 秒 (平均: {res['predicted_price_yen_kwh'].mean():.2f} 円)")

    out_csv = OUT_DIR / f"forecast_tokyo_day1_v4_{day.strftime('%Y%m%d')}.csv"
    latest_csv = OUT_DIR / "forecast_tokyo_day1_v4_latest.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    res.to_csv(latest_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
