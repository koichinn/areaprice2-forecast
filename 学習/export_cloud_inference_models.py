#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事前学習済み軽量モデル群の一括エクスポートスクリプト v4 (Grand Ensemble)
- 全8エリアモデル (models/area_models_8area.joblib):
    各エリアについて [HGB + 5物理ブロック LightGBM MAE + 5物理ブロック LightGBM q65] の
    Grand Ensemble v4 モデル群を保存。
- 東京DAY1 Grand Ensemble v4 モデル (models/tokyo_day1_v4_models.joblib):
    東京専用の物理ブロック＋分位点回帰モデル群を保存。
"""
from __future__ import annotations
import joblib, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb

sys.path.insert(0, r"P:\エリアプライス2\学習")
import evaluate_area_premium_day1_clean_rolling as E
import predict_area_day1_recon as PAR
import predict_tokyo_day1_v4 as PTV4
import build_fast_eval_cache_tokyo_day1 as B

ROOT = Path(r"P:\エリアプライス2")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BLOCKS_PHYSICAL = [
    [0, 1, 2, 3, 4, 5, 6],       # 0: Night baseline
    [7, 8],                      # 1: Morning ramp
    [9, 10, 11, 12, 13, 14],     # 2: Solar peak
    [15, 16, 17, 18, 19],        # 3: Evening lighting ramp & thermal peak
    [20, 21, 22, 23],            # 4: Late night decay
]


def train_lgb_blocks(Xtr, htr, ytr, objective="mae", alpha=0.65, n_est=180, lr=0.045, leaf=31, seed=42):
    """5物理ブロックごとの LightGBM モデル辞書を学習して返す"""
    valid = np.isfinite(ytr)
    block_models = {}
    for b_idx, hs in enumerate(BLOCKS_PHYSICAL):
        m = np.isin(htr, hs) & valid
        if m.sum() > 300:
            mdl = lgb.LGBMRegressor(
                objective=objective,
                alpha=alpha if objective == "quantile" else 0.5,
                n_estimators=n_est,
                learning_rate=lr,
                num_leaves=leaf,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                n_jobs=-1,
                random_state=seed,
                verbose=-1
            )
            mdl.fit(Xtr[m], ytr[m])
            block_models[b_idx] = mdl
        else:
            block_models[b_idx] = float(np.nanmedian(ytr[valid]))
    return block_models


def export_8area_models():
    t0 = time.time()
    AREAS = ["hokkaido", "tohoku", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]
    day = pd.Timestamp.today().normalize()
    models = {}
    print(f"=== 全8エリア Grand Ensemble v4 モデルの事前学習＆保存開始 ({day.strftime('%Y-%m-%d')}) ===")
    for ar in AREAS:
        t_ar = time.time()
        A, acol, ca, P, pcol, cp = PAR.prep(ar)
        F = PAR.fit_area(A, acol, ca, P, cp, day, 2)
        HP = PAR.HP

        # 時間帯 (時)
        dt_a = A.index.to_series().reset_index(drop=True)
        htr_a = dt_a.dt.hour.to_numpy()[F.tr_a]
        htr_p = dt_a.dt.hour.to_numpy()[F.tr_p]

        # 1. HGB (全体)
        m_a = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=HP.max_iter, learning_rate=HP.learning_rate,
            max_leaf_nodes=HP.max_leaf_nodes, min_samples_leaf=HP.min_samples_leaf, l2_regularization=HP.l2,
            early_stopping=True, validation_fraction=0.1, random_state=HP.random_state
        )
        res_a = F.ya[F.tr_a] - F.alag[F.tr_a]
        m_a.fit(F.aX[F.tr_a], res_a)

        m_p = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=HP.max_iter, learning_rate=HP.learning_rate,
            max_leaf_nodes=HP.max_leaf_nodes, min_samples_leaf=HP.min_samples_leaf, l2_regularization=HP.l2,
            early_stopping=True, validation_fraction=0.1, random_state=HP.random_state
        )
        m_p.fit(F.pX[F.tr_p], F.yp[F.tr_p])

        # 2. LightGBM 5物理ブロック (MAE & q65)
        lgb_dir_mae = train_lgb_blocks(F.aX[F.tr_a], htr_a, res_a, objective="mae", seed=42)
        lgb_dir_q65 = train_lgb_blocks(F.aX[F.tr_a], htr_a, res_a, objective="quantile", alpha=0.65, seed=42)

        lgb_prem_mae = train_lgb_blocks(F.pX[F.tr_p], htr_p, F.yp[F.tr_p], objective="mae", seed=42)
        lgb_prem_q65 = train_lgb_blocks(F.pX[F.tr_p], htr_p, F.yp[F.tr_p], objective="quantile", alpha=0.65, seed=42)

        models[ar] = {
            "direct_model": m_a,
            "premium_model": m_p,
            "direct_lgb_mae": lgb_dir_mae,
            "direct_lgb_q65": lgb_dir_q65,
            "premium_lgb_mae": lgb_prem_mae,
            "premium_lgb_q65": lgb_prem_q65,
            "ca": ca,
            "cp": cp,
            "acol": acol,
            "anchor_type": "anc_cal" if ar in PAR.CAL_ANCHOR_AREAS else "lag1d",
            "blend_w": PAR.AREA_BLEND_W.get(ar, 0.5),
            "trained_at": day.strftime("%Y-%m-%d"),
            "n_direct": F.n_a,
            "n_premium": F.n_p,
            "version": "v4_grand_ensemble",
        }
        print(f"  OK {ar:8s}: Grand Ensemble v4 学習完了 ({time.time() - t_ar:.1f}s)")

    out_path = MODELS_DIR / "area_models_8area.joblib"
    joblib.dump(models, out_path, compress=3)
    print(f"[完了] 8エリア Grand Ensemble v4 モデル保存: {out_path.stat().st_size / (1024*1024):.2f} MB ({time.time() - t0:.1f}s)")


def _fit_hgb_blocks_dict(Xtr, htr, rtr, blocks, loss="absolute_error", quantile=None, iter_scale=0.35, leaf=31, lr=0.045, it=450, msl=35, seed=0):
    fitted = {}
    valid_tr = np.isfinite(rtr)
    covered = set()
    for hs in blocks:
        m = np.isin(htr, hs) & valid_tr
        if m.sum() > 400:
            mdl = B.mk_model(loss=loss, quantile=quantile, iter_scale=iter_scale, seed=seed, leaf=leaf, lr=lr, it=it, msl=msl)
            mdl.fit(Xtr[m], rtr[m])
            fitted[tuple(hs)] = mdl
            covered.update(hs)
    if len(covered) < 24:
        mdl = B.mk_model(loss=loss, quantile=quantile, iter_scale=iter_scale, seed=seed, leaf=leaf, lr=lr, it=it, msl=msl)
        mdl.fit(Xtr[valid_tr], rtr[valid_tr])
        fitted["all"] = mdl
    return fitted


def _fit_tokyo_lgb_dict(Xtr, htr, rtr, blocks, objective="mae", alpha=0.5, n_est=180, lr=0.045, leaf=31, seed=42):
    fitted = {}
    valid_tr = np.isfinite(rtr)
    for hs in blocks:
        m = np.isin(htr, hs) & valid_tr
        if m.sum() > 400:
            mdl = lgb.LGBMRegressor(
                objective=objective,
                alpha=alpha,
                n_estimators=n_est,
                learning_rate=lr,
                num_leaves=leaf,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
            )
            mdl.fit(Xtr[m], rtr[m])
            fitted[tuple(hs)] = mdl
    return fitted


def export_tokyo_models(iter_scale: float = 0.35):
    t0 = time.time()
    print("\n=== 東京DAY1 Grand Ensemble v4 モデルの事前学習＆保存開始 ===")
    d, aux, cand_cols, must_keep = B.load_and_engineer_tokyo(extra=None)
    t = pd.to_datetime(d["target_datetime"])

    l1_t_raw = pd.to_numeric(d["area_price_yen_kwh_lag1d"], errors="coerce").fillna(20.0).to_numpy(float)
    l1_s_raw = pd.to_numeric(d["システムプライス(円/kWh)_lag1d"], errors="coerce").fillna(20.0).to_numpy(float)
    l1_p_raw = l1_t_raw - l1_s_raw
    d["_tmp_l1_tok"] = l1_t_raw
    d["_tmp_l1_sys"] = l1_s_raw
    d["_tmp_l1_prem"] = l1_p_raw
    d["_date"] = t.dt.normalize()

    daily_summary = d.groupby("_date").agg({"_tmp_l1_tok": "mean", "_tmp_l1_sys": "mean", "_tmp_l1_prem": "mean"}).sort_index()
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

    tr = d["target"].notna().to_numpy(bool)
    htr = t.dt.hour.to_numpy()[tr]
    y_tok = d["target"].to_numpy(float)
    y_sys = d["y_sys"].to_numpy(float)
    y_prem = d["y_prem"].to_numpy(float)
    y_z07 = y_tok - 0.70 * y_sys

    res_tok_cal = y_tok - aux["anc_cal_tok"]
    res_tok_sim = y_tok - aux["anc_sim_tok"]
    res_tok_lag1 = y_tok - d["area_price_yen_kwh_lag1d"].to_numpy(float)
    res_prem_cal = y_prem - aux["anc_cal_prem"]
    res_z07_cal = y_z07 - aux["anc_cal_z07"]
    res_z07_sim = y_z07 - aux["anc_sim_z07"]

    wq, wa, tok_feats_saved, prem_feats_saved, w_lgb_day, w_lgb_night = PTV4.load_calib()
    if tok_feats_saved and all(c in d.columns for c in tok_feats_saved):
        tok_feats = tok_feats_saved
    else:
        tok_feats = B.select_top_features(d, cand_cols, res_tok_cal, must_keep, top_k=195)
    tok_feats = sorted(list(set(tok_feats) | set(new_drift_feats)))

    if prem_feats_saved and all(c in d.columns for c in prem_feats_saved):
        prem_feats = prem_feats_saved
    else:
        prem_feats = B.select_top_features(d, cand_cols, res_prem_cal, must_keep, top_k=195)

    X_tok = d.loc[tr, tok_feats].to_numpy(dtype=np.float32)
    X_prem = d.loc[tr, prem_feats].to_numpy(dtype=np.float32)

    DAY_SPLIT = [B.DAY, [h for h in range(24) if h not in B.DAY]]
    WIDE_SPLIT = [B.WIDE, [h for h in range(24) if h not in B.WIDE]]

    print(f"  東京 HGB (20系統) & LightGBM (4系統) 学習中 (n_train={int(tr.sum()):,}, tok_feats={len(tok_feats)}, prem_feats={len(prem_feats)})...")
    hgb_models = {
        "pred_tok_cal_day": _fit_hgb_blocks_dict(X_tok, htr, res_tok_cal[tr], DAY_SPLIT, iter_scale=iter_scale, seed=10),
        "pred_tok_cal_wide": _fit_hgb_blocks_dict(X_tok, htr, res_tok_cal[tr], WIDE_SPLIT, iter_scale=iter_scale, seed=11),
        "pred_tok_cal_wide_deep": _fit_hgb_blocks_dict(X_tok, htr, res_tok_cal[tr], WIDE_SPLIT, iter_scale=iter_scale, leaf=47, lr=0.035, it=520, msl=28, seed=12),
        "pred_tok_cal_blk4": _fit_hgb_blocks_dict(X_tok, htr, res_tok_cal[tr], B.BLOCKS4, iter_scale=iter_scale, seed=13),
        "pred_tok_cal_blk6": _fit_hgb_blocks_dict(X_tok, htr, res_tok_cal[tr], B.BLOCKS6, iter_scale=iter_scale, seed=14),
        "pred_tok_cal_q65": _fit_hgb_blocks_dict(X_tok, htr, res_tok_cal[tr], WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=15),
        "pred_tok_sim_wide": _fit_hgb_blocks_dict(X_tok, htr, res_tok_sim[tr], WIDE_SPLIT, iter_scale=iter_scale, seed=17),
        "pred_tok_sim_blk6": _fit_hgb_blocks_dict(X_tok, htr, res_tok_sim[tr], B.BLOCKS6, iter_scale=iter_scale, seed=18),
        "pred_tok_sim_q65": _fit_hgb_blocks_dict(X_tok, htr, res_tok_sim[tr], WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=19),
        "pred_tok_lag1_wide": _fit_hgb_blocks_dict(X_tok, htr, res_tok_lag1[tr], WIDE_SPLIT, iter_scale=iter_scale, seed=20),
        "pred_tok_lag1_blk6": _fit_hgb_blocks_dict(X_tok, htr, res_tok_lag1[tr], B.BLOCKS6, iter_scale=iter_scale, seed=21),
        "pred_prem_cal_wide": _fit_hgb_blocks_dict(X_prem, htr, res_prem_cal[tr], WIDE_SPLIT, iter_scale=iter_scale, seed=30),
        "pred_prem_cal_blk6": _fit_hgb_blocks_dict(X_prem, htr, res_prem_cal[tr], B.BLOCKS6, iter_scale=iter_scale, seed=31),
        "pred_prem_cal_q65": _fit_hgb_blocks_dict(X_prem, htr, res_prem_cal[tr], WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=32),
        "pred_z07_cal_wide": _fit_hgb_blocks_dict(X_prem, htr, res_z07_cal[tr], WIDE_SPLIT, iter_scale=iter_scale, seed=40),
        "pred_z07_cal_blk6": _fit_hgb_blocks_dict(X_prem, htr, res_z07_cal[tr], B.BLOCKS6, iter_scale=iter_scale, seed=41),
        "pred_z07_cal_q65": _fit_hgb_blocks_dict(X_prem, htr, res_z07_cal[tr], WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=42),
        "pred_z07_sim_wide": _fit_hgb_blocks_dict(X_prem, htr, res_z07_sim[tr], WIDE_SPLIT, iter_scale=iter_scale, seed=43),
        "pred_z07_sim_blk6": _fit_hgb_blocks_dict(X_prem, htr, res_z07_sim[tr], B.BLOCKS6, iter_scale=iter_scale, seed=44),
        "pred_z07_sim_q65": _fit_hgb_blocks_dict(X_prem, htr, res_z07_sim[tr], WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=45),
    }

    lgb_models = {
        "q60": _fit_tokyo_lgb_dict(X_tok, htr, res_tok_cal[tr], PTV4.BLOCKS_PHYSICAL, objective="quantile", alpha=0.60, seed=101),
        "q65": _fit_tokyo_lgb_dict(X_tok, htr, res_tok_cal[tr], PTV4.BLOCKS_PHYSICAL, objective="quantile", alpha=0.65, seed=102),
        "sim_q65": _fit_tokyo_lgb_dict(X_tok, htr, res_tok_sim[tr], PTV4.BLOCKS_PHYSICAL, objective="quantile", alpha=0.65, seed=103),
        "l1": _fit_tokyo_lgb_dict(X_tok, htr, res_tok_cal[tr], PTV4.BLOCKS_PHYSICAL, objective="mae", alpha=0.50, seed=104),
    }

    tokyo_models = {
        "hgb_models": hgb_models,
        "lgb_models": lgb_models,
        "tok_feats": tok_feats,
        "prem_feats": prem_feats,
        "wq": wq,
        "wa": wa,
        "w_lgb_day": w_lgb_day,
        "w_lgb_night": w_lgb_night,
        "iter_scale": iter_scale,
        "n_train": int(tr.sum()),
        "trained_at": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "version": "v4_tokyo_grand_ensemble_bit_exact",
    }
    out_tokyo = MODELS_DIR / "tokyo_day1_v4_models.joblib"
    joblib.dump(tokyo_models, out_tokyo, compress=3)
    print(f"[完了] 東京 Grand Ensemble v4 モデル保存: {out_tokyo.stat().st_size / (1024*1024):.2f} MB ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-8area", action="store_true", help="8エリアモデルのみ再学習・保存する")
    args = ap.parse_args()
    export_8area_models()
    if not args.only_8area:
        export_tokyo_models()
