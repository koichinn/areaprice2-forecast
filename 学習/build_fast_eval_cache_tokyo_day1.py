#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tokyo Area Price DAY1 高速検証＆予測キャッシュ基盤 v4
- Path A (Tokyo Direct): カレンダー同期アンカー(anc_cal_tok) + 予報類似度アンカー(anc_sim_tok) + 安定レジームアンカー(anc_regime_tok)
- Path B (Reconstructed = System v4_sub2_super + Tokyo Premium): anc_cal_prem / anc_sim_prem / 直接premium
- Path C (Partial-Coupling Reconstructed = 0.70 * System v4_sub2_super + z0.70): 東西部分結合アンカー(anc_cal_z07 / anc_sim_z07)
- 特徴量: 東京気象(58本) + 全国・9エリア気象(75本) + 東西気象差 + OCCTO全国(da8) + OCCTOエリア別・東西需給差・連系線(FC/相馬双葉) + JEPX入札逼迫・近隣エリア価格ラグ
全6 foldウォークフォワード検証(gap_days=1, リークゼロ)の予測値と補助列を oof_cache_tokyo_day1.parquet に保存する。
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import jpholiday
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, r"P:\エリアプライス2\学習")
import evaluate_system_day1_levers as ev
import predict_system_day1_occto as pr
import fast_tune_system_day1 as ft

V8_DIR = Path(r"P:\エリアプライス2\database\統合データ\training_v8_price_regime")
TOK_A_PATH = V8_DIR / "jepx_training_v8_price_regime_tokyo_day1.parquet"
TOK_P_PATH = V8_DIR / "jepx_training_v8_tokyo_premium_day1.parquet"
SYS_NAT_PATH = V8_DIR / "jepx_training_v8_price_regime_system_natwf_day1.parquet"
DA8_PATH = Path(r"P:\エリアプライス2\学習\evaluation_system_day1_levers_da\occto_da8.parquet")
OCC_AIC_PATH = Path(r"P:\エリアプライス2\学習\evaluation_system_day1_levers_da\occto_area_ic_da.parquet")
JEPX_MASTER = Path(r"P:\エリアプライス2\database\JPEXデータ\data\jepx_spot_summary_master.csv")
SYS_CACHE_PATH = Path(r"P:\エリアプライス2\学習\fast_eval_cache\oof_cache_system_day1.parquet")
OUT_DIR = Path(r"P:\エリアプライス2\学習\fast_eval_cache")
OUT_FILE = OUT_DIR / "oof_cache_tokyo_day1.parquet"

WIDE = list(range(7, 20))
DAY = ev.DAY
BLOCKS4 = [[21, 22, 23, 0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16], [17, 18, 19, 20]]
BLOCKS6 = [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22, 23]]

NOACT = r"^(?!システムプライス|area_price_yen_kwh|売り入札量|買い入札量|約定総量|pr_|premium_|dt_|l\d+_|fc_sp_|soma_sp_|chu_sys_).*_lag\d+d$"


def weighted_quantile_mat(vals_mat: np.ndarray, weights_mat: np.ndarray, q: float, fallback: np.ndarray) -> np.ndarray:
    v = np.where(weights_mat > 0, vals_mat, np.inf)
    order = np.argsort(v, axis=0)
    v_sorted = np.take_along_axis(vals_mat, order, axis=0)
    w_sorted = np.take_along_axis(weights_mat, order, axis=0)
    cum_w = np.cumsum(w_sorted, axis=0)
    idx_q = np.argmax(cum_w >= q, axis=0)
    res = v_sorted[idx_q, np.arange(vals_mat.shape[1])]
    return np.where(weights_mat.sum(axis=0) > 1e-9, res, fallback)


def compute_system_v4_oof() -> pd.DataFrame:
    sys_oof = pd.read_parquet(SYS_CACHE_PATH).sort_values("target_datetime").reset_index(drop=True)
    dt = pd.to_datetime(sys_oof["target_datetime"])
    cm = ft.get_calendar_masks(dt)
    dow = dt.dt.dayofweek.to_numpy()
    hour = dt.dt.hour.to_numpy()
    sys_oof["delta_ens3_lag1"] = 0.25 * sys_oof["pred_delta_day_lag1"] + 0.45 * sys_oof["pred_delta_wide_lag1"] + 0.30 * sys_oof["pred_delta_blk_lag1"]
    sys_oof["delta_ens3_mon"] = 0.25 * sys_oof["pred_delta_day_mon"] + 0.45 * sys_oof["pred_delta_wide_mon"] + 0.30 * sys_oof["pred_delta_blk_mon"]
    mon_daytime = (dow == 0) & (hour >= 7) & (hour <= 21)
    sys_oof["delta_hybrid_mon75"] = np.where(mon_daytime, 0.75 * sys_oof["delta_ens3_mon"] + 0.25 * sys_oof["delta_ens3_lag1"], sys_oof["delta_ens3_lag1"])
    sys_oof["delta_qblend_mon75"] = ft.wf_quantile_asymmetric_blend(sys_oof, sys_oof["delta_hybrid_mon75"].to_numpy(float), "pred_q65_mon", "pred_q35_mon")
    anc_hybrid = np.where(mon_daytime, 0.75 * sys_oof["lag7"] + 0.25 * sys_oof["lag1"], sys_oof["lag1"])
    sys_oof["delta_full_v12"] = ft.wf_hourly_shrink(sys_oof, sys_oof["delta_qblend_mon75"].to_numpy(float), anc_hybrid)
    sys_oof["v3_cal_ens5"] = (
        0.15 * sys_oof["pred_v3_cal_day"] +
        0.25 * sys_oof["pred_v3_cal_wide"] +
        0.20 * sys_oof["pred_v3_cal_wide_deep"] +
        0.15 * sys_oof["pred_v3_cal_blk4"] +
        0.25 * sys_oof["pred_v3_cal_blk6"]
    )
    sys_oof["v3_cal_corr"] = ft.wf_anchor_dilution_correction(sys_oof, sys_oof["v3_cal_ens5"].to_numpy(float), cm)
    v4_res = ft.build_v4_sub2_predictions(sys_oof, cm)
    for k, v in v4_res.items():
        sys_oof[k] = v
    return sys_oof


def load_and_engineer_tokyo(extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str], set[str]]:
    """extra: 本番予測用の追加行 (直近の文脈行 + DAY1の48コマ)。列 is_extra=True を持ち、学習には使わない。"""
    # 1. Load Tokyo Area v8 and Tokyo Premium v8 (drop built-in tomorrow_today occto cols to prevent leakage)
    da = pd.read_parquet(TOK_A_PATH)
    dp = pd.read_parquet(TOK_P_PATH)
    da["target_datetime"] = pd.to_datetime(da["target_datetime"])
    dp["target_datetime"] = pd.to_datetime(dp["target_datetime"])
    da = da[[c for c in da.columns if "occto" not in c.lower() and not c.startswith("pr_interact_")]].sort_values("target_datetime").reset_index(drop=True)
    p_only = ["target_datetime", "target"] + [c for c in dp.columns if c.startswith("premium_")]
    dp_sub = dp[p_only].rename(columns={"target": "y_prem"}).sort_values("target_datetime").reset_index(drop=True)
    d = da.merge(dp_sub, on="target_datetime", how="inner")

    # 2. Merge National & 9-area weather forecasts from system_natwf_day1
    sn = pd.read_parquet(SYS_NAT_PATH)
    sn["target_datetime"] = pd.to_datetime(sn["target_datetime"])
    sn_wf_cols = [c for c in sn.columns if c.startswith("wf_") and c not in d.columns]
    sn_sub = sn[["target_datetime", "target"] + sn_wf_cols].rename(columns={"target": "y_sys"})
    d = d.merge(sn_sub, on="target_datetime", how="inner")

    # 3. Merge clean dayaftertomorrow OCCTO (da8 + area/IC)
    da8 = pd.read_parquet(DA8_PATH)
    da8["target_datetime"] = pd.to_datetime(da8["target_datetime"])
    da8 = da8.groupby("target_datetime", as_index=False)[pr.OCC5].mean()
    d = d.merge(da8, on="target_datetime", how="left")

    aic = pd.read_parquet(OCC_AIC_PATH)
    aic["target_datetime"] = pd.to_datetime(aic["target_datetime"])
    aic_cols = [c for c in aic.columns if c != "target_datetime" and c not in d.columns]
    d = d.merge(aic[["target_datetime"] + aic_cols], on="target_datetime", how="left")
    d["is_extra"] = False
    if extra is not None and len(extra):
        ex = extra.copy()
        ex["target_datetime"] = pd.to_datetime(ex["target_datetime"])
        ex = ex[~ex["target_datetime"].isin(d["target_datetime"])]
        ex["is_extra"] = True
        d = pd.concat([d, ex[[c for c in ex.columns if c in d.columns]]], ignore_index=True, sort=False)
        d["is_extra"] = d["is_extra"].fillna(False).astype(bool)
        d = d.sort_values("target_datetime").reset_index(drop=True)

    # 4. Merge confirmed neighbor area price lags (D-1, D-2, D-7, D-14) from JEPX Master
    jm = pd.read_csv(JEPX_MASTER, encoding="utf-8-sig", low_memory=False)
    jm["dt"] = pd.to_datetime(jm["受渡日"]) + pd.to_timedelta((pd.to_numeric(jm["時刻コード"]) - 1) * 30, unit="m")
    jm = jm.drop_duplicates("dt", keep="last").set_index("dt")
    t_ser = pd.to_datetime(d["target_datetime"])
    nb_cols = {}
    for jp, en in [("東北", "toh"), ("中部", "chu"), ("関西", "kan"), ("北海道", "hok"), ("九州", "kyu")]:
        s_area = pd.to_numeric(jm[f"エリアプライス{jp}(円/kWh)"], errors="coerce")
        for k_lag in (1, 2, 7, 14):
            nb_cols[f"l{k_lag}_price_{en}"] = s_area.reindex(t_ser - pd.Timedelta(days=k_lag)).to_numpy(float)

    l1_tok = pd.to_numeric(d["area_price_yen_kwh_lag1d"], errors="coerce").to_numpy(float)
    l2_tok = pd.to_numeric(d["area_price_yen_kwh_lag2d"], errors="coerce").to_numpy(float)
    l7_tok = pd.to_numeric(d["area_price_yen_kwh_lag7d"], errors="coerce").to_numpy(float)
    l14_tok = pd.to_numeric(d["area_price_yen_kwh_lag14d"], errors="coerce").to_numpy(float)

    l1_sys = pd.to_numeric(d["システムプライス(円/kWh)_lag1d"], errors="coerce").to_numpy(float)
    l2_sys = pd.to_numeric(d["システムプライス(円/kWh)_lag2d"], errors="coerce").to_numpy(float)
    l7_sys = pd.to_numeric(d["システムプライス(円/kWh)_lag7d"], errors="coerce").to_numpy(float)
    l14_sys = pd.to_numeric(d["システムプライス(円/kWh)_lag14d"], errors="coerce").to_numpy(float)

    for k_lag, lt, ls in [(1, l1_tok, l1_sys), (2, l2_tok, l2_sys), (7, l7_tok, l7_sys), (14, l14_tok, l14_sys)]:
        nb_cols[f"fc_sp_lag{k_lag}d"] = lt - nb_cols[f"l{k_lag}_price_chu"]
        nb_cols[f"soma_sp_lag{k_lag}d"] = lt - nb_cols[f"l{k_lag}_price_toh"]
        nb_cols[f"chu_sys_sp_lag{k_lag}d"] = nb_cols[f"l{k_lag}_price_chu"] - ls
        nb_cols[f"fc_split_flag_lag{k_lag}d"] = (np.abs(nb_cols[f"fc_sp_lag{k_lag}d"]) >= 0.02).astype(float)

    d = pd.concat([d, pd.DataFrame(nb_cols, index=d.index)], axis=1)

    # Filter to valid rows with lag1, lag7, lag14
    valid_mask = (d["target"].notna() & d["area_price_yen_kwh_lag1d"].notna() & d["area_price_yen_kwh_lag7d"].notna() & d["area_price_yen_kwh_lag14d"].notna()) | d["is_extra"]
    d = d[valid_mask].sort_values("target_datetime").reset_index(drop=True)

    t = pd.to_datetime(d["target_datetime"])
    cm = pr.get_calendar_masks(t)
    dow = cm["dow"]
    hour = cm["hour"]
    is_hol = cm["is_hol"]
    is_rst = cm["is_rst"]
    rst_obon = cm["rst_obon"]
    prev1_hol = cm["prev1_hol"]
    prev2_hol = cm["prev2_hol"]
    prev7_hol = cm["prev7_hol"]
    prev1_rest = cm["prev1_rest"]
    prev2_rest = cm["prev2_rest"]
    daytime = cm["daytime"]

    # Calendar-synced weights (w1, w2, w7, w14)
    w1 = np.ones(len(d), dtype=float)
    w2 = np.zeros(len(d), dtype=float)
    w7 = np.zeros(len(d), dtype=float)
    w14 = np.zeros(len(d), dtype=float)

    m_hol_mon = is_hol & (dow == 0); w1[m_hol_mon], w7[m_hol_mon] = 0.55, 0.45
    m_hol_wk = is_hol & (dow >= 1) & (~prev1_rest) & prev2_rest; w1[m_hol_wk], w2[m_hol_wk] = 0.25, 0.75
    m_hol_mid = is_hol & (dow >= 1) & (~prev1_rest) & (~prev2_rest); w1[m_hol_mid], w7[m_hol_mid] = 0.50, 0.50
    m_mon_p7hol = (~is_hol) & (dow == 0) & prev7_hol & daytime; w1[m_mon_p7hol], w14[m_mon_p7hol] = 0.25, 0.75
    m_mon_norm = (~is_hol) & (dow == 0) & (~prev7_hol) & daytime; w1[m_mon_norm], w7[m_mon_norm] = 0.25, 0.75
    m_tue_ahol = (~is_hol) & (dow == 1) & prev1_hol & daytime; w1[m_tue_ahol], w7[m_tue_ahol] = 0.45, 0.55
    m_wed_ahol = (~is_hol) & (dow >= 2) & (dow <= 4) & prev1_hol & daytime; w1[m_wed_ahol], w2[m_wed_ahol], w7[m_wed_ahol] = 0.40, 0.25, 0.35
    m_sat = (~is_hol) & (dow == 5) & daytime; w1[m_sat], w7[m_sat], w14[m_sat] = 0.55, 0.25, 0.20

    l1_t = d["area_price_yen_kwh_lag1d"].to_numpy(float)
    l2_t = d["area_price_yen_kwh_lag2d"].to_numpy(float)
    l7_t = d["area_price_yen_kwh_lag7d"].to_numpy(float)
    l14_t = d["area_price_yen_kwh_lag14d"].to_numpy(float)

    l1_s = d["システムプライス(円/kWh)_lag1d"].to_numpy(float)
    l2_s = d["システムプライス(円/kWh)_lag2d"].to_numpy(float)
    l7_s = d["システムプライス(円/kWh)_lag7d"].to_numpy(float)
    l14_s = d["システムプライス(円/kWh)_lag14d"].to_numpy(float)

    l1_p = np.where(d["premium_lag1d"].notna(), pd.to_numeric(d["premium_lag1d"], errors="coerce"), l1_t - l1_s) if "premium_lag1d" in d.columns else (l1_t - l1_s)
    l2_p = np.where(d["premium_lag2d"].notna(), pd.to_numeric(d["premium_lag2d"], errors="coerce"), l2_t - l2_s) if "premium_lag2d" in d.columns else (l2_t - l2_s)
    l7_p = np.where(d["premium_lag7d"].notna(), pd.to_numeric(d["premium_lag7d"], errors="coerce"), l7_t - l7_s) if "premium_lag7d" in d.columns else (l7_t - l7_s)
    l14_p = np.where(d["premium_lag14d"].notna(), pd.to_numeric(d["premium_lag14d"], errors="coerce"), l14_t - l14_s) if "premium_lag14d" in d.columns else (l14_t - l14_s)

    anc_cal_tok = w1 * l1_t + w2 * l2_t + w7 * l7_t + w14 * l14_t
    anc_cal_sys = w1 * l1_s + w2 * l2_s + w7 * l7_s + w14 * l14_s
    anc_cal_prem = w1 * l1_p + w2 * l2_p + w7 * l7_p + w14 * l14_p
    anc_cal_z07 = anc_cal_tok - 0.70 * anc_cal_sys

    lag_stack_tok = np.vstack([l1_t, l2_t, l7_t, l14_t])
    sorted_lags_tok = np.sort(lag_stack_tok, axis=0)
    anc_regime_tok = np.where(is_rst, 0.5 * (sorted_lags_tok[0] + sorted_lags_tok[1]), np.nanmedian(np.vstack([anc_cal_tok, l7_t, l14_t]), axis=0))

    # East-West Weather & OCCTO Nonlinear Proxies
    sol_tok = pd.to_numeric(d["wf_tokyo_solar_effective_wm2"], errors="coerce").fillna(pd.to_numeric(d["wf_shortwave_radiation_wm2"], errors="coerce")).fillna(0.0).to_numpy(float)
    sol_toh = pd.to_numeric(d["wf_tohoku_solar_effective_wm2"], errors="coerce").fillna(pd.Series(sol_tok, index=d.index)).to_numpy(float)
    sol_chu = pd.to_numeric(d["wf_chubu_solar_effective_wm2"], errors="coerce").fillna(pd.Series(sol_tok, index=d.index)).to_numpy(float)
    sol_kan = pd.to_numeric(d["wf_kansai_solar_effective_wm2"], errors="coerce").fillna(pd.Series(sol_tok, index=d.index)).to_numpy(float)
    sol_kyu = pd.to_numeric(d["wf_kyushu_solar_effective_wm2"], errors="coerce").fillna(pd.Series(sol_tok, index=d.index)).to_numpy(float)
    sol_nat = pd.to_numeric(d["wf_nat_solar_effective_wm2_mean"], errors="coerce").fillna(pd.Series(sol_tok, index=d.index)).to_numpy(float)

    sol_east50 = 0.75 * sol_tok + 0.25 * sol_toh
    sol_west60 = 0.35 * sol_chu + 0.35 * sol_kan + 0.30 * sol_kyu
    temp_tok = pd.to_numeric(d["wf_temperature_c"], errors="coerce").fillna(20.0).to_numpy(float)
    temp_nat = pd.to_numeric(d["wf_nat_temperature_c_mean"], errors="coerce").fillna(pd.Series(temp_tok, index=d.index)).to_numpy(float)

    util_nat = pd.to_numeric(d["pre1100_occto_wide_utilization_pct"], errors="coerce").fillna(84.0).to_numpy(float)
    res_nat = pd.to_numeric(d["pre1100_occto_wide_reserve_rate_pct"], errors="coerce").fillna(14.0).to_numpy(float)
    util_tok_w = pd.to_numeric(d["occ_tok_wide_util"], errors="coerce").fillna(pd.Series(util_nat, index=d.index)).to_numpy(float)
    res_tok_w = pd.to_numeric(d["occ_tok_wide_res_rate"], errors="coerce").fillna(pd.Series(res_nat, index=d.index)).to_numpy(float)
    util_tok_a = pd.to_numeric(d["occ_tok_area_util"], errors="coerce").fillna(pd.Series(util_tok_w, index=d.index)).to_numpy(float)
    dem_tok_a = pd.to_numeric(d["occ_tok_area_demand"], errors="coerce").fillna(30000.0).to_numpy(float)
    dem_east50 = pd.to_numeric(d["occ_east50_demand"], errors="coerce").fillna(38000.0).to_numpy(float)
    util_ew_diff = pd.to_numeric(d["occ_diff_east_west_util"], errors="coerce").fillna(0.0).to_numpy(float)
    fc_fwd_free = pd.to_numeric(d["ic_fc_fwd_free"], errors="coerce").fillna(1200.0).to_numpy(float)
    fc_bwd_free = pd.to_numeric(d["ic_fc_bwd_free"], errors="coerce").fillna(1200.0).to_numpy(float)

    # 28-day Forecast-Similarity Kernel Anchors (strictly confirmed D-1..D-28)
    df_sim_src = pd.DataFrame({
        "y_tok": d["target"].to_numpy(float),
        "y_sys": d["y_sys"].to_numpy(float),
        "y_prem": d["y_prem"].to_numpy(float),
        "sol_east": sol_east50,
        "sol_west": sol_west60,
        "sol_nat": sol_nat,
        "util_tok_w": util_tok_w,
        "util_tok_a": util_tok_a,
        "util_ew_diff": util_ew_diff,
        "temp_tok": temp_tok,
        "fc_bwd_free": fc_bwd_free,
        "is_rst": is_rst.astype(float),
        "dow": dow.astype(float),
    }, index=t)

    K = 28
    mats = {col: np.zeros((K, len(d))) for col in df_sim_src.columns}
    for k_lag in range(1, K + 1):
        sh = df_sim_src.reindex(t - pd.Timedelta(days=k_lag))
        for col in df_sim_src.columns:
            mats[col][k_lag - 1] = sh[col].to_numpy(float)

    same_rst = (mats["is_rst"] == is_rst[None, :]) & np.isfinite(mats["y_tok"])
    same_dow = (mats["dow"] == dow[None, :]).astype(float)
    ks = np.arange(1, K + 1, dtype=float)[:, None]

    dist_tok = (
        1.2 * ((mats["sol_east"] - sol_east50[None, :]) / 135.0) ** 2
        + 1.0 * ((mats["util_tok_w"] - util_tok_w[None, :]) / 3.3) ** 2
        + 0.85 * ((mats["util_tok_a"] - util_tok_a[None, :]) / 3.8) ** 2
        + 0.40 * ((mats["temp_tok"] - temp_tok[None, :]) / 3.8) ** 2
        - 0.25 * same_dow
    )
    w_sim = np.where(same_rst, np.exp(-0.5 * dist_tok) * (0.5 ** ((ks - 1.0) / 9.5)), 0.0)
    w_sum = w_sim.sum(axis=0)
    val_sim = w_sum > 1e-9
    w_norm = np.where(val_sim[None, :], w_sim / np.where(val_sim, w_sum, 1.0)[None, :], 0.0)

    sim_tok_mean = np.where(val_sim, (w_norm * np.nan_to_num(mats["y_tok"], nan=0.0)).sum(axis=0), anc_cal_tok)
    sim_tok_q50 = weighted_quantile_mat(mats["y_tok"], w_norm, 0.50, anc_cal_tok)
    sim_tok_q65 = weighted_quantile_mat(mats["y_tok"], w_norm, 0.65, anc_cal_tok)
    sim_tok_q35 = weighted_quantile_mat(mats["y_tok"], w_norm, 0.35, anc_cal_tok)
    anc_sim_tok = 0.55 * sim_tok_mean + 0.45 * sim_tok_q50

    sim_prem_mean = np.where(val_sim, (w_norm * np.nan_to_num(mats["y_prem"], nan=0.0)).sum(axis=0), anc_cal_prem)
    sim_prem_q50 = weighted_quantile_mat(mats["y_prem"], w_norm, 0.50, anc_cal_prem)
    anc_sim_prem = 0.55 * sim_prem_mean + 0.45 * sim_prem_q50

    z07_mat = mats["y_tok"] - 0.70 * mats["y_sys"]
    sim_z07_mean = np.where(val_sim, (w_norm * np.nan_to_num(z07_mat, nan=0.0)).sum(axis=0), anc_cal_z07)
    sim_z07_q50 = weighted_quantile_mat(z07_mat, w_norm, 0.50, anc_cal_z07)
    anc_sim_z07 = 0.55 * sim_z07_mean + 0.45 * sim_z07_q50

    # Daily daytime average solar & utilization for False Mean-Reversion Shield & Regime Calibration
    s_daily_df = pd.DataFrame({"sol_e": sol_east50, "sol_w": sol_west60, "util_t": util_tok_w, "util_a": util_tok_a}, index=t)
    day_m = (hour >= 8) & (hour <= 16)
    daily_means = s_daily_df[day_m].groupby(s_daily_df[day_m].index.normalize()).mean()
    d_norm = t.dt.normalize()
    sol_e_d = d_norm.map(daily_means["sol_e"]).to_numpy(float)
    sol_w_d = d_norm.map(daily_means["sol_w"]).to_numpy(float)
    util_t_d = d_norm.map(daily_means["util_t"]).to_numpy(float)
    sol_e_l1 = (d_norm - pd.Timedelta(days=1)).map(daily_means["sol_e"]).to_numpy(float)
    sol_w_l1 = (d_norm - pd.Timedelta(days=1)).map(daily_means["sol_w"]).to_numpy(float)
    util_t_l1 = (d_norm - pd.Timedelta(days=1)).map(daily_means["util_t"]).to_numpy(float)

    # Pre-bidding nonlinear & intraday ramp features
    pre_cols = {}
    pre_cols["wf_sol_east50"] = sol_east50
    pre_cols["wf_sol_west60"] = sol_west60
    pre_cols["wf_sol_diff_east_west"] = sol_east50 - sol_west60
    pre_cols["wf_sol_diff_tok_nat"] = sol_tok - sol_nat
    pre_cols["wf_sol_diff_tok_chu"] = sol_tok - sol_chu
    pre_cols["wf_temp_diff_tok_nat"] = temp_tok - temp_nat

    pre_cols["nl_tok_res_tight12"] = np.maximum(0.0, 12.0 - res_tok_w) ** 1.5
    pre_cols["nl_tok_res_tight9"] = np.maximum(0.0, 9.0 - res_tok_w) ** 2.0
    pre_cols["nl_tok_util_high88"] = np.maximum(0.0, util_tok_w - 88.0) ** 1.5
    pre_cols["nl_tok_util_high91"] = np.maximum(0.0, util_tok_w - 91.0) ** 2.0
    pre_cols["nl_tok_area_util_high88"] = np.maximum(0.0, util_tok_a - 88.0) ** 1.5
    pre_cols["nl_east_solar_surplus"] = np.maximum(0.0, sol_east50 - 200.0) * np.maximum(0.0, 84.0 - util_tok_w)
    pre_cols["nl_west_solar_split_pressure"] = np.maximum(0.0, sol_west60 - sol_east50 + 50.0) * np.maximum(0.0, util_tok_a - 80.0) / 100.0
    pre_cols["nl_tok_net_demand"] = dem_tok_a - 25.0 * sol_east50
    pre_cols["nl_east50_net_demand"] = dem_east50 - 35.0 * sol_east50
    pre_cols["nl_fc_cong_proxy"] = np.maximum(0.0, sol_west60 - 150.0) / (np.maximum(100.0, fc_bwd_free) / 500.0)

    s_se = pd.Series(sol_east50)
    s_sw = pd.Series(sol_west60)
    s_ut = pd.Series(util_tok_w)
    s_ua = pd.Series(util_tok_a)
    s_nd = pd.Series(pre_cols["nl_tok_net_demand"])

    pre_cols["ramp1h_sol_east"] = (s_se - s_se.shift(2)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_sol_east"] = (s_se - s_se.shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_fwd_sol_east"] = (s_se.shift(-4) - s_se).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_sol_diff_ew"] = ((s_se - s_sw) - (s_se - s_sw).shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_tok_util"] = (s_ut - s_ut.shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_tok_area_util"] = (s_ua - s_ua.shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_tok_net_demand"] = (s_nd - s_nd.shift(4)).fillna(0.0).to_numpy(float)

    d = pd.concat([d, pd.DataFrame(pre_cols, index=d.index)], axis=1)

    # Compute d1_, d2_, d7_, d14_, dcal_ deltas for key weather & OCCTO columns
    delta_target_cols = [
        c for c in (
            pr.DELTA_BASE_COLS
            + list(pre_cols.keys())
            + [
                "occ_tok_area_demand", "occ_tok_area_supply", "occ_tok_area_reserve",
                "occ_tok_area_util", "occ_tok_wide_util", "occ_tok_wide_res_rate",
                "occ_east50_demand", "occ_east50_util", "occ_west60_util",
                "occ_diff_tok_chu_util", "occ_diff_east_west_util", "occ_diff_tok_chu_wide_res",
                "ic_fc_fwd_free", "ic_fc_bwd_free", "ic_fc_fwd_flow", "ic_fc_min_free",
                "ic_soma_fwd_free", "ic_soma_bwd_free",
                "wf_shortwave_radiation_wm2", "wf_direct_radiation_wm2", "wf_temperature_c", "wf_cloud_cover_pct",
            ]
        )
        if c in d.columns
    ]
    delta_target_cols = list(dict.fromkeys(delta_target_cols))
    base_df = d.set_index("target_datetime")[delta_target_cols]
    curr_v = base_df.to_numpy(float)
    v1 = base_df.reindex(t - pd.Timedelta(days=1)).to_numpy(float)
    v2 = base_df.reindex(t - pd.Timedelta(days=2)).to_numpy(float)
    v7 = base_df.reindex(t - pd.Timedelta(days=7)).to_numpy(float)
    v14 = base_df.reindex(t - pd.Timedelta(days=14)).to_numpy(float)

    d1_m = np.where(np.isnan(curr_v - v1), curr_v - v2, curr_v - v1)  # 前日行が無い場合(本番の文脈欠け)のみ2日差で代替
    d2_m = np.where(np.isnan(curr_v - v2), d1_m, curr_v - v2)
    d7_m = np.where(np.isnan(curr_v - v7), d1_m, curr_v - v7)
    d14_m = np.where(np.isnan(curr_v - v14), d7_m, curr_v - v14)
    dcal_m = (
        np.where(w1[:, None] > 0, w1[:, None] * d1_m, 0.0)
        + np.where(w2[:, None] > 0, w2[:, None] * d2_m, 0.0)
        + np.where(w7[:, None] > 0, w7[:, None] * d7_m, 0.0)
        + np.where(w14[:, None] > 0, w14[:, None] * d14_m, 0.0)
    )

    new_cols = {}
    for i, c in enumerate(delta_target_cols):
        new_cols[f"d1_{c}"] = d1_m[:, i]
        new_cols[f"d7_{c}"] = d7_m[:, i]
        new_cols[f"dcal_{c}"] = dcal_m[:, i]
        if c.startswith(("pre1100_occto_", "occ_", "nl_", "ic_")) or "solar" in c or "shortwave" in c:
            new_cols[f"d2_{c}"] = d2_m[:, i]
            new_cols[f"d14_{c}"] = d14_m[:, i]

    # Price & Premium & Similarity anchor features
    med_lag_t = np.nanmedian(lag_stack_tok, axis=0)
    min_lag_t = np.nanmin(lag_stack_tok, axis=0)
    new_cols["anc_cal_tok_level"] = anc_cal_tok
    new_cols["anc_regime_tok_level"] = anc_regime_tok
    new_cols["anc_sim_tok_level"] = anc_sim_tok
    new_cols["sim_tok_mean_level"] = sim_tok_mean
    new_cols["sim_tok_q50_level"] = sim_tok_q50
    new_cols["sim_tok_q65_level"] = sim_tok_q65
    new_cols["sim_tok_q35_level"] = sim_tok_q35
    new_cols["sim_tok_iqr"] = sim_tok_q65 - sim_tok_q35
    new_cols["anc_sim_vs_cal_tok"] = anc_sim_tok - anc_cal_tok
    new_cols["anc_sim_vs_regime_tok"] = anc_sim_tok - anc_regime_tok
    new_cols["anc_cal_vs_regime_tok"] = anc_cal_tok - anc_regime_tok
    new_cols["anc_cal_vs_med_lag_tok"] = anc_cal_tok - med_lag_t
    new_cols["anc_cal_vs_min_lag_tok"] = anc_cal_tok - min_lag_t
    new_cols["lag1_vs_med_lag_tok"] = l1_t - med_lag_t
    new_cols["lag1_vs_lag7_tok"] = l1_t - l7_t
    new_cols["lag1_vs_lag2_tok"] = l1_t - l2_t

    new_cols["anc_cal_prem_level"] = anc_cal_prem
    new_cols["anc_sim_prem_level"] = anc_sim_prem
    new_cols["sim_prem_mean_level"] = sim_prem_mean
    new_cols["sim_prem_q50_level"] = sim_prem_q50
    new_cols["anc_sim_vs_cal_prem"] = anc_sim_prem - anc_cal_prem
    new_cols["anc_cal_z07_level"] = anc_cal_z07
    new_cols["anc_sim_z07_level"] = anc_sim_z07

    # JEPX Bid Imbalance features
    sell_l1 = pd.to_numeric(d["売り入札量(kWh)_lag1d"], errors="coerce").to_numpy(float)
    buy_l1 = pd.to_numeric(d["買い入札量(kWh)_lag1d"], errors="coerce").to_numpy(float)
    sell_l2 = pd.to_numeric(d["売り入札量(kWh)_lag2d"], errors="coerce").to_numpy(float)
    buy_l2 = pd.to_numeric(d["買い入札量(kWh)_lag2d"], errors="coerce").to_numpy(float)
    sell_l7 = pd.to_numeric(d["売り入札量(kWh)_lag7d"], errors="coerce").to_numpy(float)
    buy_l7 = pd.to_numeric(d["買い入札量(kWh)_lag7d"], errors="coerce").to_numpy(float)

    imb_l1 = np.nan_to_num((buy_l1 - sell_l1) / np.maximum(1.0, sell_l1), nan=0.0)
    imb_l2 = np.nan_to_num((buy_l2 - sell_l2) / np.maximum(1.0, sell_l2), nan=0.0)
    imb_l7 = np.nan_to_num((buy_l7 - sell_l7) / np.maximum(1.0, sell_l7), nan=0.0)
    new_cols["bid_imb_l1"] = imb_l1
    new_cols["bid_imb_l7"] = imb_l7
    new_cols["bid_imb_diff_1_7"] = imb_l1 - imb_l7
    new_cols["bid_imb_diff_1_2"] = imb_l1 - imb_l2
    new_cols["sell_drop_ratio_1_7"] = np.nan_to_num((sell_l1 - sell_l7) / np.maximum(1.0, sell_l7), nan=0.0)

    # Calendar flags
    new_cols["cal_is_hol"] = is_hol.astype(float)
    new_cols["cal_is_rest"] = is_rst.astype(float)
    new_cols["cal_is_obon"] = rst_obon.astype(float)
    new_cols["cal_prev1_hol"] = prev1_hol.astype(float)
    new_cols["cal_prev2_hol"] = prev2_hol.astype(float)
    new_cols["cal_prev7_hol"] = prev7_hol.astype(float)
    new_cols["cal_prev1_rest"] = prev1_rest.astype(float)
    new_cols["cal_prev2_rest"] = prev2_rest.astype(float)
    new_cols["cal_rst_tomorrow"] = cm["rst_tomorrow"].astype(float)
    new_cols["cal_dow"] = dow.astype(float)

    # Nonlinear interaction terms
    if "dcal_occ_tok_wide_util" in new_cols:
        new_cols["ix_anc_dev_x_dcal_tok_util"] = (anc_cal_tok - med_lag_t) * new_cols["dcal_occ_tok_wide_util"]
    if "dcal_wf_sol_east50" in new_cols:
        new_cols["ix_anc_dev_x_dcal_sol_east"] = (anc_cal_tok - med_lag_t) * new_cols["dcal_wf_sol_east50"]
        new_cols["ix_anc_level_x_dcal_sol_east"] = anc_cal_tok * new_cols["dcal_wf_sol_east50"]
    if "dcal_wf_sol_diff_east_west" in new_cols:
        new_cols["ix_prem_anc_x_dcal_sol_ew"] = anc_cal_prem * new_cols["dcal_wf_sol_diff_east_west"]

    d_out = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)

    aux = {
        "anc_cal_tok": anc_cal_tok,
        "anc_regime_tok": anc_regime_tok,
        "anc_sim_tok": anc_sim_tok,
        "sim_tok_mean": sim_tok_mean,
        "sim_tok_q50": sim_tok_q50,
        "sim_tok_q65": sim_tok_q65,
        "sim_tok_q35": sim_tok_q35,
        "anc_cal_prem": anc_cal_prem,
        "anc_sim_prem": anc_sim_prem,
        "sim_prem_mean": sim_prem_mean,
        "sim_prem_q50": sim_prem_q50,
        "anc_cal_z07": anc_cal_z07,
        "anc_sim_z07": anc_sim_z07,
        "sim_z07_mean": sim_z07_mean,
        "sim_z07_q50": sim_z07_q50,
        "anc_cal_sys": anc_cal_sys,
        "sol_e_d": np.nan_to_num(sol_e_d, nan=0.0),
        "sol_w_d": np.nan_to_num(sol_w_d, nan=0.0),
        "util_t_d": np.nan_to_num(util_t_d, nan=84.0),
        "d1_sol_e_d": np.nan_to_num(sol_e_d - sol_e_l1, nan=0.0),
        "d1_sol_w_d": np.nan_to_num(sol_w_d - sol_w_l1, nan=0.0),
        "d1_util_t_d": np.nan_to_num(util_t_d - util_t_l1, nan=0.0),
        "bid_imb_diff_1_7": new_cols["bid_imb_diff_1_7"],
        "rst_yesterday": cm["rst_yesterday"],
        "rst_tomorrow": cm["rst_tomorrow"],
    }

    # Candidate feature pool (strictly excluding target, y_sys, y_prem, metadata, and actual weather/supply lags)
    exclude_exact = {
        "as_of_datetime", "target_datetime", "datetime", "area", "target_name",
        "model_segment", "forecast_horizon_days", "information_rule",
        "target", "y_sys", "y_prem", "システムプライス(円/kWh)", "area_price_yen_kwh", "is_extra",
    } | pr.EXCLUDE_FEATURES

    cand_cols = []
    for c in d_out.columns:
        if c in exclude_exact or c.startswith("post1100_occto_"):
            continue
        if re.search(NOACT, c):
            continue
        if pd.api.types.is_numeric_dtype(d_out[c]) and d_out[c].notna().sum() >= 500:
            cand_cols.append(c)

    must_keep = (
        set(pr.OCC5)
        | set(pre_cols.keys())
        | {c for c in new_cols if c.startswith(("cal_", "anc_", "sim_", "lag1_vs_", "bid_", "ix_", "dcal_pre1100_", "dcal_occ_", "dcal_nl_", "dcal_wf_sol_"))}
        | {c for c in nb_cols.keys()}
    ) & set(cand_cols)

    return d_out, aux, cand_cols, must_keep


def select_top_features(d: pd.DataFrame, feats: list[str], y_res: np.ndarray, must_keep: set[str], top_k: int = 195) -> list[str]:
    scores = {}
    y_f = pd.Series(y_res).fillna(0.0).to_numpy(float)
    for c in feats:
        x = pd.to_numeric(d[c], errors="coerce")
        if x.notna().sum() < 500 or x.nunique() < 2:
            scores[c] = 0.0
            continue
        x_f = x.fillna(x.median()).to_numpy(float)
        r = np.corrcoef(x_f, y_f)[0, 1]
        scores[c] = float(abs(r)) if np.isfinite(r) else 0.0
    ranked = sorted(feats, key=lambda col: (10.0 if col in must_keep else 0.0) + scores.get(col, 0.0), reverse=True)
    return ranked[:top_k]


def mk_model(loss="absolute_error", quantile=None, iter_scale=1.0, seed=0, leaf=31, lr=0.045, it=450, msl=35):
    kw = dict(
        loss=loss, max_iter=max(10, int(it * iter_scale)), learning_rate=lr,
        max_leaf_nodes=leaf, min_samples_leaf=msl, l2_regularization=1.2,
        early_stopping=False, random_state=seed
    )
    if loss == "quantile" and quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def blocks_fit_pred(Xtr, htr, rtr, Xte, hte, blocks, loss="absolute_error", quantile=None, iter_scale=1.0, leaf=31, lr=0.045, it=450, msl=35, seed=0):
    p = np.zeros(len(Xte))
    used = np.zeros(len(p), bool)
    valid_tr = np.isfinite(rtr)
    for hs in blocks:
        m = np.isin(htr, hs) & valid_tr
        s = np.isin(hte, hs)
        if s.any() and m.sum() > 400:
            mdl = mk_model(loss=loss, quantile=quantile, iter_scale=iter_scale, seed=seed, leaf=leaf, lr=lr, it=it, msl=msl)
            mdl.fit(Xtr[m], rtr[m])
            p[s] = mdl.predict(Xte[s])
            used |= s
    if not used.all():
        m_all = valid_tr
        mdl = mk_model(loss=loss, quantile=quantile, iter_scale=iter_scale, seed=seed, leaf=leaf, lr=lr, it=it, msl=msl)
        mdl.fit(Xtr[m_all], rtr[m_all])
        p[~used] = mdl.predict(Xte[~used])
    return p


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-scale", type=float, default=0.55)
    ap.add_argument("--top-k", type=int, default=195)
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("1. System DAY1 v4_sub2_super (MAE=1.9991) キャッシュを読み込み中...", flush=True)
    sys_oof = compute_system_v4_oof()
    sys_map = sys_oof.set_index(pd.to_datetime(sys_oof["target_datetime"]))

    print("2. 東京エリア特徴量 + 全国/9エリア気象 + OCCTOエリア/連系線 + 類似度アンカーを生成中...", flush=True)
    d, aux, cand_cols, must_keep = load_and_engineer_tokyo()
    t = pd.to_datetime(d["target_datetime"])
    hour = t.dt.hour.to_numpy()

    y_tok = d["target"].to_numpy(float)
    y_sys = d["y_sys"].to_numpy(float)
    y_prem = d["y_prem"].to_numpy(float)
    y_z07 = y_tok - 0.70 * y_sys

    l1_t = d["area_price_yen_kwh_lag1d"].to_numpy(float)
    l2_t = d["area_price_yen_kwh_lag2d"].to_numpy(float)
    l7_t = d["area_price_yen_kwh_lag7d"].to_numpy(float)
    l14_t = d["area_price_yen_kwh_lag14d"].to_numpy(float)
    l1_p = d["premium_lag1d"].to_numpy(float)
    l7_p = d["premium_lag7d"].to_numpy(float)

    anc_cal_tok = aux["anc_cal_tok"]
    anc_reg_tok = aux["anc_regime_tok"]
    anc_sim_tok = aux["anc_sim_tok"]
    anc_cal_prem = aux["anc_cal_prem"]
    anc_sim_prem = aux["anc_sim_prem"]
    anc_cal_z07 = aux["anc_cal_z07"]
    anc_sim_z07 = aux["anc_sim_z07"]

    res_tok_cal = y_tok - anc_cal_tok
    res_tok_sim = y_tok - anc_sim_tok
    res_tok_lag1 = y_tok - l1_t
    res_tok_lag7 = y_tok - l7_t
    res_tok_reg = y_tok - anc_reg_tok

    res_prem_cal = y_prem - anc_cal_prem
    res_prem_sim = y_prem - anc_sim_prem
    res_z07_cal = y_z07 - anc_cal_z07
    res_z07_sim = y_z07 - anc_sim_z07

    tok_feats = select_top_features(d, cand_cols, res_tok_cal, must_keep, top_k=a.top_k)
    prem_feats = select_top_features(d, cand_cols, res_prem_cal, must_keep, top_k=a.top_k)
    print(f"   選択特徴量: Direct={len(tok_feats)}列, Premium/Coupled={len(prem_feats)}列 (必須保持={len(must_keep)}列)", flush=True)

    X_tok = d[tok_feats].to_numpy(dtype=np.float32)
    X_prem = d[prem_feats].to_numpy(dtype=np.float32)

    DAY_SPLIT = [DAY, [h for h in range(24) if h not in DAY]]
    WIDE_SPLIT = [WIDE, [h for h in range(24) if h not in WIDE]]

    oof_list = []
    # Match the exact 6 folds of sys_oof
    for k in sorted(sys_oof["fold_idx"].unique()):
        f_t0 = time.time()
        g_sys = sys_oof[sys_oof["fold_idx"] == k]
        fold_name = str(g_sys["fold"].iloc[0])
        f_dts = pd.to_datetime(g_sys["target_datetime"])
        te_start = f_dts.min()
        te = t.isin(f_dts).to_numpy()
        tr = (t < te_start.normalize() - pd.Timedelta(days=1)).to_numpy()
        htr, hte = hour[tr], hour[te]

        # --- Path A: Direct Tokyo Models ---
        p_tok_cal_day = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, DAY_SPLIT, iter_scale=a.iter_scale, seed=10)
        p_tok_cal_wide = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=11)
        p_tok_cal_wide_deep = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, leaf=47, lr=0.035, it=520, msl=28, seed=12)
        p_tok_cal_blk4 = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, BLOCKS4, iter_scale=a.iter_scale, seed=13)
        p_tok_cal_blk6 = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=14)
        p_tok_cal_q65 = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=a.iter_scale, seed=15)
        p_tok_cal_q35 = anc_cal_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.35, iter_scale=a.iter_scale, seed=16)

        p_tok_sim_wide = anc_sim_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=17)
        p_tok_sim_blk6 = anc_sim_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=18)
        p_tok_sim_q65 = anc_sim_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=a.iter_scale, seed=19)

        p_tok_lag1_wide = l1_t[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_lag1[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=20)
        p_tok_lag1_blk6 = l1_t[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_lag1[tr], X_tok[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=21)
        p_tok_lag7_wide = l7_t[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_lag7[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=22)
        p_tok_reg_wide = anc_reg_tok[te] + blocks_fit_pred(X_tok[tr], htr, res_tok_reg[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=23)

        # --- Path B: Tokyo Premium Models ---
        p_prem_cal_wide = anc_cal_prem[te] + blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=30)
        p_prem_cal_blk6 = anc_cal_prem[te] + blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=31)
        p_prem_cal_q65 = anc_cal_prem[te] + blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=a.iter_scale, seed=32)
        p_prem_cal_q35 = anc_cal_prem[te] + blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.35, iter_scale=a.iter_scale, seed=33)

        p_prem_sim_wide = anc_sim_prem[te] + blocks_fit_pred(X_prem[tr], htr, res_prem_sim[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=34)
        p_prem_sim_blk6 = anc_sim_prem[te] + blocks_fit_pred(X_prem[tr], htr, res_prem_sim[tr], X_prem[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=35)
        p_prem_raw_wide = blocks_fit_pred(X_prem[tr], htr, y_prem[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=36)
        p_prem_raw_blk6 = blocks_fit_pred(X_prem[tr], htr, y_prem[tr], X_prem[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=37)

        # --- Path C: Partial-Coupling z07 = y_tok - 0.70 * y_sys Models ---
        p_z07_cal_wide = anc_cal_z07[te] + blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=40)
        p_z07_cal_blk6 = anc_cal_z07[te] + blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=41)
        p_z07_cal_q65 = anc_cal_z07[te] + blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=a.iter_scale, seed=42)
        p_z07_sim_wide = anc_sim_z07[te] + blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=a.iter_scale, seed=43)
        p_z07_sim_blk6 = anc_sim_z07[te] + blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, BLOCKS6, iter_scale=a.iter_scale, seed=44)
        p_z07_sim_q65 = anc_sim_z07[te] + blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=a.iter_scale, seed=45)

        dt_te = t[te]
        sys_sub = sys_map.reindex(dt_te)
        p_sys_v4 = sys_sub["v4_sub2_super"].to_numpy(float)
        p_sys_ewma = sys_sub["v4_causal_ewma"].to_numpy(float)
        p_sys_base = sys_sub["v4_prior_q_base"].to_numpy(float)

        f_df = pd.DataFrame({
            "fold_idx": k,
            "fold": fold_name,
            "target_datetime": dt_te.to_numpy(),
            "y": y_tok[te],
            "y_sys": y_sys[te],
            "y_prem": y_prem[te],
            "lag1": l1_t[te],
            "lag2": l2_t[te],
            "lag7": l7_t[te],
            "lag14": l14_t[te],
            "prem_lag1": l1_p[te],
            "prem_lag7": l7_p[te],
            "anc_cal": anc_cal_tok[te],
            "anc_regime": anc_reg_tok[te],
            "anc_sim": anc_sim_tok[te],
            "sim_tok_mean": aux["sim_tok_mean"][te],
            "sim_tok_q50": aux["sim_tok_q50"][te],
            "sim_tok_q65": aux["sim_tok_q65"][te],
            "sim_tok_q35": aux["sim_tok_q35"][te],
            "anc_cal_prem": anc_cal_prem[te],
            "anc_sim_prem": anc_sim_prem[te],
            "anc_cal_z07": anc_cal_z07[te],
            "anc_sim_z07": anc_sim_z07[te],
            "anc_cal_sys": aux["anc_cal_sys"][te],
            "anc_sim_sys": sys_sub["anc_sim"].to_numpy(float),
            "p_sys_v4": p_sys_v4,
            "p_sys_ewma": p_sys_ewma,
            "p_sys_base": p_sys_base,
            # Path A predictions
            "pred_tok_cal_day": p_tok_cal_day,
            "pred_tok_cal_wide": p_tok_cal_wide,
            "pred_tok_cal_wide_deep": p_tok_cal_wide_deep,
            "pred_tok_cal_blk4": p_tok_cal_blk4,
            "pred_tok_cal_blk6": p_tok_cal_blk6,
            "pred_tok_cal_q65": p_tok_cal_q65,
            "pred_tok_cal_q35": p_tok_cal_q35,
            "pred_tok_sim_wide": p_tok_sim_wide,
            "pred_tok_sim_blk6": p_tok_sim_blk6,
            "pred_tok_sim_q65": p_tok_sim_q65,
            "pred_tok_lag1_wide": p_tok_lag1_wide,
            "pred_tok_lag1_blk6": p_tok_lag1_blk6,
            "pred_tok_lag7_wide": p_tok_lag7_wide,
            "pred_tok_reg_wide": p_tok_reg_wide,
            # Path B predictions (premium)
            "pred_prem_cal_wide": p_prem_cal_wide,
            "pred_prem_cal_blk6": p_prem_cal_blk6,
            "pred_prem_cal_q65": p_prem_cal_q65,
            "pred_prem_cal_q35": p_prem_cal_q35,
            "pred_prem_sim_wide": p_prem_sim_wide,
            "pred_prem_sim_blk6": p_prem_sim_blk6,
            "pred_prem_raw_wide": p_prem_raw_wide,
            "pred_prem_raw_blk6": p_prem_raw_blk6,
            # Path C predictions (z07 = y_tok - 0.70 * y_sys)
            "pred_z07_cal_wide": p_z07_cal_wide,
            "pred_z07_cal_blk6": p_z07_cal_blk6,
            "pred_z07_cal_q65": p_z07_cal_q65,
            "pred_z07_sim_wide": p_z07_sim_wide,
            "pred_z07_sim_blk6": p_z07_sim_blk6,
            "pred_z07_sim_q65": p_z07_sim_q65,
            # Auxiliary regime columns
            "sol_e_d": aux["sol_e_d"][te],
            "sol_w_d": aux["sol_w_d"][te],
            "util_t_d": aux["util_t_d"][te],
            "d1_sol_e_d": aux["d1_sol_e_d"][te],
            "d1_sol_w_d": aux["d1_sol_w_d"][te],
            "d1_util_t_d": aux["d1_util_t_d"][te],
            "bid_imb_diff_1_7": aux["bid_imb_diff_1_7"][te],
            "rst_yesterday": aux["rst_yesterday"][te],
            "rst_tomorrow": aux["rst_tomorrow"][te],
        })
        oof_list.append(f_df)

        y_te = y_tok[te]
        dir_ens = 0.20 * p_tok_cal_wide + 0.20 * p_tok_cal_blk6 + 0.30 * p_tok_sim_wide + 0.30 * p_tok_sim_blk6
        rec_ens = p_sys_v4 + 0.25 * p_prem_cal_wide + 0.25 * p_prem_cal_blk6 + 0.25 * p_prem_sim_wide + 0.25 * p_prem_sim_blk6
        z07_ens = 0.70 * p_sys_v4 + 0.25 * p_z07_cal_wide + 0.25 * p_z07_cal_blk6 + 0.25 * p_z07_sim_wide + 0.25 * p_z07_sim_blk6
        mix3 = 0.35 * dir_ens + 0.25 * rec_ens + 0.40 * z07_ens
        print(
            f"  [Fold {k+1}/6 {fold_name}] Direct={np.mean(np.abs(y_te - dir_ens)):.3f} | "
            f"Recon={np.mean(np.abs(y_te - rec_ens)):.3f} | "
            f"CoupledZ07={np.mean(np.abs(y_te - z07_ens)):.3f} | "
            f"Mix3={np.mean(np.abs(y_te - mix3)):.3f} ({time.time()-f_t0:.1f}s)",
            flush=True
        )

    out_df = pd.concat(oof_list, ignore_index=True)
    out_df.to_parquet(OUT_FILE, index=False)
    print(f"\n=== 東京DAY1 キャッシュ保存完了: {OUT_FILE} ({len(out_df):,}行, {out_df.shape[1]}列, 総時間 {time.time()-t0:.1f}s) ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
