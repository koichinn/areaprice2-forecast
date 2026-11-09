#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System DAY1 超高速チューニング＆検証エンジン v6 (v4 Sub-2.00 Super-Ensemble 対応)
- v3 カレンダー同期5構造アンサンブル + v4 予報類似度アンカー(anc_sim)・JEPX入札逼迫モデル + v2 デルタモデル
- スケール不変アンカー希釈補正 + 曜日・時間帯別 Prior Quantile ブレンド + 時間帯別シュリンケージ
- 偽平均回帰シールド (False Mean-Reversion Shield) + Walk-Forward 需給・境界レジーム較正 + 因果的 Dual EWMA 残差フィードバック
oof_cache_system_day1.parquet を読み込み、約1秒で全6 foldのウォークフォワード検証(リークゼロ)を行う。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import jpholiday

CACHE_DIR = Path(r"P:\エリアプライス2\学習\fast_eval_cache")
CACHE_FILE = CACHE_DIR / "oof_cache_system_day1.parquet"
BLOCKS6 = [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22, 23]]


def get_calendar_masks(dts: pd.Series) -> dict[str, np.ndarray]:
    dates = dts.dt.date
    dow = dts.dt.dayofweek.to_numpy()
    hour = dts.dt.hour.to_numpy()
    min_d = pd.Timestamp(dates.min()) - pd.Timedelta(days=20)
    max_d = pd.Timestamp(dates.max()) + pd.Timedelta(days=5)
    all_d = pd.date_range(min_d, max_d, freq="D").date
    hol_set = {
        x for x in all_d
        if jpholiday.is_holiday(x)
        or (x.month == 8 and x.day in (13, 14, 15))
        or (x.month == 1 and x.day in (2, 3))
        or (x.month == 12 and x.day in (29, 30, 31))
    }

    def is_rest(x):
        w = pd.Timestamp(x).dayofweek
        return (w >= 5) or (x in hol_set)

    is_hol = np.array([x in hol_set for x in dates], dtype=bool)
    is_rst = np.array([is_rest(x) for x in dates], dtype=bool)
    prev1_hol = np.array([(pd.Timestamp(x) - pd.Timedelta(days=1)).date() in hol_set for x in dates], dtype=bool)
    prev7_hol = np.array([(pd.Timestamp(x) - pd.Timedelta(days=7)).date() in hol_set for x in dates], dtype=bool)
    prev1_rest = np.array([is_rest((pd.Timestamp(x) - pd.Timedelta(days=1)).date()) for x in dates], dtype=bool)
    prev2_rest = np.array([is_rest((pd.Timestamp(x) - pd.Timedelta(days=2)).date()) for x in dates], dtype=bool)
    daytime = (hour >= 7) & (hour <= 21)

    return {
        "dow": dow,
        "hour": hour,
        "is_hol": is_hol,
        "is_rst": is_rst,
        "prev1_hol": prev1_hol,
        "prev7_hol": prev7_hol,
        "prev1_rest": prev1_rest,
        "prev2_rest": prev2_rest,
        "daytime": daytime,
        "m_mon_p7hol": (~is_hol) & (dow == 0) & prev7_hol & daytime,
        "m_mon_norm": (~is_hol) & (dow == 0) & (~prev7_hol) & daytime,
        "m_tue_ahol": (~is_hol) & (dow == 1) & prev1_hol & daytime,
        "m_hol_wk": is_hol & (dow >= 1) & (~prev1_rest) & prev2_rest,
    }


def wf_anchor_dilution_correction(df: pd.DataFrame, base_arr: np.ndarray, cm: dict[str, np.ndarray]) -> np.ndarray:
    """カレンダー遷移日における休日/平日アンカー希釈の比例補正係数 gamma を過去fold(folds < k)から学習(リークなし)"""
    folds = df["fold_idx"].to_numpy()
    y = df["y"].to_numpy(float)
    l1 = df["lag1"].to_numpy(float)
    l2 = df["lag2"].to_numpy(float)
    l7 = df["lag7"].to_numpy(float)
    l14 = df["lag14"].to_numpy(float)
    v3_lag7 = df["pred_v3_lag7_wide"].to_numpy(float)
    v3_reg = df["pred_v3_reg_wide"].to_numpy(float)

    delta_vec = np.zeros(len(df), dtype=float)
    m1 = cm["m_mon_p7hol"]
    m2 = cm["m_mon_norm"]
    m3 = cm["m_tue_ahol"]
    m4 = cm["m_hol_wk"]

    delta_vec[m1] = np.maximum(0.0, l14[m1] - l1[m1])
    delta_vec[m2] = 0.50 * np.maximum(0.0, l7[m2] - l1[m2]) + 0.50 * (v3_lag7[m2] - base_arr[m2])
    delta_vec[m3] = 0.50 * np.maximum(0.0, l7[m3] - l1[m3]) + 0.50 * (v3_lag7[m3] - base_arr[m3])
    delta_vec[m4] = 0.50 * (l2[m4] - l1[m4]) + 0.50 * (v3_reg[m4] - base_arr[m4])

    out = base_arr.copy()
    gammas = np.linspace(0.0, 0.40, 17)
    trans_mask = m1 | m2 | m3 | m4

    for k in sorted(np.unique(folds)):
        if k < 2:
            out[(folds == k) & m1] += 0.20 * delta_vec[(folds == k) & m1]
            continue
        past = (folds < k) & trans_mask
        curr = (folds == k) & trans_mask
        if past.sum() < 20 or not curr.any():
            continue
        y_p = y[past]
        b_p = base_arr[past]
        d_p = delta_vec[past]
        best_g = 0.0
        best_mae = float(np.mean(np.abs(y_p - b_p)))
        for g in gammas:
            mae = float(np.mean(np.abs(y_p - (b_p + g * d_p))))
            if mae < best_mae:
                best_mae = mae
                best_g = g
        out[curr] = base_arr[curr] + best_g * delta_vec[curr]
    return out


def wf_quantile_asymmetric_blend(df: pd.DataFrame, base_arr: np.ndarray, q65_col: str, q35_col: str) -> np.ndarray:
    """時間帯ごとに base_arr と Quantile(q65/q35) の最適ブレンド比率を過去foldから学習(リークなし)"""
    dt = pd.to_datetime(df["target_datetime"])
    hour = dt.dt.hour.to_numpy()
    folds = df["fold_idx"].to_numpy()
    y = df["y"].to_numpy(float)
    q65 = df[q65_col].to_numpy(float)
    q35 = df[q35_col].to_numpy(float)

    out = base_arr.copy()
    for k in sorted(np.unique(folds)):
        if k == 0:
            continue
        past = folds < k
        curr = folds == k
        for h in range(24):
            h_pool = [(h - 1) % 24, h, (h + 1) % 24] if k == 1 else [h]
            m_past = past & np.isin(hour, h_pool)
            m_curr = curr & (hour == h)
            if m_past.sum() < 20 or not m_curr.any():
                continue
            y_p = y[m_past]
            b_p = base_arr[m_past]
            q65_p = q65[m_past]
            q35_p = q35[m_past]

            best_w65, best_w35 = 0.0, 0.0
            best_mae = float(np.mean(np.abs(y_p - b_p)))
            for w65 in [0.0, 0.06, 0.12, 0.20, 0.28, 0.35]:
                for w35 in [0.0, 0.06, 0.12, 0.20, 0.28]:
                    if w65 + w35 > 0.50:
                        continue
                    cand = (1.0 - w65 - w35) * b_p + w65 * q65_p + w35 * q35_p
                    mae = float(np.mean(np.abs(y_p - cand)))
                    if mae < best_mae - 0.001:
                        best_mae = mae
                        best_w65, best_w35 = w65, w35

            out[m_curr] = (1.0 - best_w65 - best_w35) * base_arr[m_curr] + best_w65 * q65[m_curr] + best_w35 * q35[m_curr]
    return out


def wf_hourly_shrink(df: pd.DataFrame, base_arr: np.ndarray, anc_arr: np.ndarray) -> np.ndarray:
    """時間帯ごとに残差シュリンク率 alpha_h を過去foldから学習して適用(リークなし)"""
    dt = pd.to_datetime(df["target_datetime"])
    hour = dt.dt.hour.to_numpy()
    folds = df["fold_idx"].to_numpy()
    y = df["y"].to_numpy(float)
    res = base_arr - anc_arr

    out = base_arr.copy()
    alphas = np.linspace(0.3, 1.25, 20)
    for k in sorted(np.unique(folds)):
        if k == 0:
            continue
        past = folds < k
        curr = folds == k
        for h in range(24):
            h_pool = [(h - 1) % 24, h, (h + 1) % 24] if k == 1 else [h]
            m_past = past & np.isin(hour, h_pool)
            m_curr = curr & (hour == h)
            if m_past.sum() < 20 or not m_curr.any():
                continue
            y_p = y[m_past]
            anc_p = anc_arr[m_past]
            res_p = res[m_past]
            best_a = 1.0
            best_mae = float(np.mean(np.abs(y_p - (anc_p + res_p))))
            for a in alphas:
                mae = float(np.mean(np.abs(y_p - (anc_p + a * res_p))))
                if mae < best_mae - 0.001:
                    best_mae = mae
                    best_a = a
            out[m_curr] = anc_arr[m_curr] + best_a * res[m_curr]
    return out


def wf_learn_median_offset(y: np.ndarray, pred: np.ndarray, folds: np.ndarray, mask_cond: np.ndarray, prior_offset: float, min_samples: int = 120, shrink_n: float = 500.0) -> np.ndarray:
    """過去fold (folds < k) の中央値誤差と事前オフセットの経験ベイズ結合(リークなし)"""
    out = pred.copy()
    for k in sorted(np.unique(folds)):
        te = (folds == k) & mask_cond
        if not te.any():
            continue
        tr = (folds < k) & mask_cond
        n = int(tr.sum())
        if k <= 1 or n < min_samples:
            off = prior_offset
        else:
            emp_med = float(np.median(y[tr] - pred[tr]))
            w = n / (n + shrink_n)
            off = w * emp_med + (1.0 - w) * prior_offset
        out[te] = pred[te] + off
    return out


def build_v4_sub2_predictions(df: pd.DataFrame, cm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """リークゼロで v4_sub2_super (Weighted MAE < 2.0000) の段階別予測を構築"""
    dt = pd.to_datetime(df["target_datetime"])
    dow = cm["dow"]
    hour = cm["hour"]
    is_rest = cm["is_rst"]
    folds = df["fold_idx"].to_numpy()
    y = df["y"].to_numpy(float)
    anc_cal = df["anc_cal"].to_numpy(float)
    anc_reg = df["anc_regime"].to_numpy(float)
    anc_sim = df["anc_sim"].to_numpy(float)
    delta_v2 = df["delta_full_v12"].to_numpy(float)

    v3_cal_corr = df["v3_cal_corr"].to_numpy(float)
    q65_corr = wf_anchor_dilution_correction(df, df["pred_v3_cal_q65"].to_numpy(float), cm)

    if "pred_v4_sim_wide_l1" in df.columns and "pred_v4_sim_wide_q65" in df.columns:
        v4_l1_corr = wf_anchor_dilution_correction(df, df["pred_v4_sim_wide_l1"].to_numpy(float), cm)
        v4_q65_corr = wf_anchor_dilution_correction(df, df["pred_v4_sim_wide_q65"].to_numpy(float), cm)
        w_v4 = 0.20
        l1_mix = (1.0 - w_v4) * v3_cal_corr + w_v4 * v4_l1_corr
        q65_mix = (1.0 - w_v4) * q65_corr + w_v4 * v4_q65_corr
        anc_mix = (1.0 - w_v4) * anc_cal + w_v4 * anc_sim
    else:
        w_v4 = 0.0
        l1_mix = v3_cal_corr
        q65_mix = q65_corr
        anc_mix = anc_cal

    is_peak = (hour >= 15) & (hour <= 21)
    is_midday = (hour >= 10) & (hour <= 14)
    w_q = np.where(is_rest, 0.36, np.where(is_peak, 0.78, np.where(is_midday, 0.52, 0.55)))
    base_q = (1.0 - w_q) * l1_mix + w_q * q65_mix
    shrunk_q = wf_hourly_shrink(df, base_q, anc_mix)
    p_base = 0.80 * shrunk_q + 0.20 * delta_v2

    p_sim = p_base + (0.25 * (1.0 - w_v4)) * (anc_sim - anc_cal)
    d1_sol = df["d1_sol_d"].to_numpy(float)
    d1_util = df["d1_util_d"].to_numpy(float)
    sol_cur = df["sol_d"].to_numpy(float)
    util_cur = df["util_d"].to_numpy(float)

    high_regime_persist = ((anc_sim - anc_reg) > 2.0) & (d1_sol <= 20.0) & (d1_util >= -1.5)
    res_sim_p = p_sim - anc_sim
    p_guard = np.where(high_regime_persist & (res_sim_p < 0), anc_sim + 0.55 * res_sim_p, p_sim)

    tight_surge = (((util_cur >= 88.0) & (d1_util >= 3.5) & (sol_cur <= 220.0)) | ((util_cur >= 90.0) & (d1_sol <= -60.0) & (d1_util >= 2.5))) & (hour >= 8) & (hour <= 21)
    ext_sol_drop = (d1_sol <= -180.0) & (df["lag1"].to_numpy(float) >= 24.0) & (hour >= 8) & (hour <= 21)

    p_g3 = wf_learn_median_offset(y, p_guard, folds, tight_surge, prior_offset=+1.25, min_samples=120, shrink_n=500.0)
    p_g3 = wf_learn_median_offset(y, p_g3, folds, ext_sol_drop, prior_offset=-1.70, min_samples=120, shrink_n=500.0)

    # 厳密因果的 Dual EWMA 残差フィードバック (d_p < d_cur の確定実績のみ使用)
    dates_norm = dt.dt.normalize()
    unique_dates = sorted(dates_norm.unique())
    blk_id = np.zeros(len(df), dtype=int)
    for bi, hs in enumerate(BLOCKS6):
        blk_id[np.isin(hour, hs)] = bi

    df_tmp = pd.DataFrame({"date": dates_norm, "blk": blk_id, "err": y - p_g3, "is_rest": is_rest})
    blk_daily = df_tmp.groupby(["date", "blk"])["err"].median().unstack("blk")
    date_is_rest = df_tmp.groupby("date")["is_rest"].first()

    ewma_all_arr = np.zeros(len(df), dtype=float)
    ewma_same_arr = np.zeros(len(df), dtype=float)
    for d_cur in unique_dates:
        m_cur = (dates_norm == d_cur).to_numpy()
        cur_rst = bool(date_is_rest.loc[d_cur])
        past_dates = [d_p for d_p in unique_dates if d_p < d_cur and (d_cur - d_p).days <= 21]
        if not past_dates:
            continue
        ages = np.array([(d_cur - d_p).days for d_p in past_dates], dtype=float)
        w_all = 0.5 ** ((ages - 1.0) / 6.0)
        w_all /= w_all.sum()
        same_mask = np.array([bool(date_is_rest.loc[d_p]) == cur_rst for d_p in past_dates], dtype=bool)
        for bi in range(len(BLOCKS6)):
            mb = m_cur & (blk_id == bi)
            if not mb.any():
                continue
            vals = np.array([blk_daily.loc[d_p, bi] for d_p in past_dates], dtype=float)
            ewma_all_arr[mb] = float(np.dot(w_all, vals))
            if same_mask.any():
                w_s = 0.5 ** ((ages[same_mask] - 1.0) / 6.0)
                w_s /= w_s.sum()
                ewma_same_arr[mb] = float(np.dot(w_s, vals[same_mask]))
            else:
                ewma_same_arr[mb] = ewma_all_arr[mb]

    p_ewma = p_g3 + 0.20 * np.clip(ewma_all_arr, -3.5, 3.5) + 0.15 * np.clip(ewma_same_arr, -3.5, 3.5)

    imb_diff = df["bid_imb_diff_1_7"].to_numpy(float)
    rst_yesterday = df["rst_yesterday"].to_numpy(bool)
    rst_tomorrow = df["rst_tomorrow"].to_numpy(bool)

    m_scarcity = (~is_rest) & (imb_diff >= 0.08) & (d1_util >= 0.0) & (hour >= 8) & (hour <= 21)
    m_abundance = (imb_diff <= -0.08) & (d1_sol >= 0.0) & (d1_util <= 0.0) & (hour >= 8) & (hour <= 21)
    m_mon_surge = (dow == 0) & (~is_rest) & (d1_util >= 6.0) & (hour >= 7) & (hour <= 21)
    m_sun_eve = is_rest & (~rst_tomorrow) & (hour >= 16) & (hour <= 23)
    m_mon_morn = (~is_rest) & rst_yesterday & (hour >= 0) & (hour <= 7)
    m_sat_morn = is_rest & (~rst_yesterday) & (hour >= 0) & (hour <= 7)
    m_tue_tight = (dow == 1) & (~is_rest) & (util_cur >= 86.0) & (d1_util >= 1.0) & (hour >= 8) & (hour <= 21)

    p_final = wf_learn_median_offset(y, p_ewma, folds, m_scarcity, prior_offset=+0.70, min_samples=120, shrink_n=500.0)
    p_final = wf_learn_median_offset(y, p_final, folds, m_abundance, prior_offset=-0.35, min_samples=120, shrink_n=500.0)
    p_final = wf_learn_median_offset(y, p_final, folds, m_mon_surge, prior_offset=+0.75, min_samples=120, shrink_n=500.0)
    p_final = wf_learn_median_offset(y, p_final, folds, m_sun_eve, prior_offset=+0.65, min_samples=120, shrink_n=500.0)
    p_final = wf_learn_median_offset(y, p_final, folds, m_mon_morn, prior_offset=-0.40, min_samples=120, shrink_n=500.0)
    p_final = wf_learn_median_offset(y, p_final, folds, m_sat_morn, prior_offset=+0.45, min_samples=120, shrink_n=500.0)
    p_final = wf_learn_median_offset(y, p_final, folds, m_tue_tight, prior_offset=+0.50, min_samples=120, shrink_n=500.0)

    return {
        "v4_prior_q_base": p_base,
        "v4_guard_shield": p_g3,
        "v4_causal_ewma": p_ewma,
        "v4_sub2_super": p_final,
    }


def main():
    t0 = time.time()
    df = pd.read_parquet(CACHE_FILE).sort_values("target_datetime").reset_index(drop=True)
    dt = pd.to_datetime(df["target_datetime"])
    dow = dt.dt.dayofweek.to_numpy()
    hour = dt.dt.hour.to_numpy()
    cm = get_calendar_masks(dt)
    is_rest = cm["is_rst"]

    # --- v2 モデル ---
    if "pred_delta_day_lag1" in df.columns:
        df["delta_ens3_lag1"] = 0.25 * df["pred_delta_day_lag1"] + 0.45 * df["pred_delta_wide_lag1"] + 0.30 * df["pred_delta_blk_lag1"]
        df["delta_ens3_mon"] = 0.25 * df["pred_delta_day_mon"] + 0.45 * df["pred_delta_wide_mon"] + 0.30 * df["pred_delta_blk_mon"]
        mon_daytime = (dow == 0) & (hour >= 7) & (hour <= 21)
        df["delta_hybrid_mon75"] = np.where(mon_daytime, 0.75 * df["delta_ens3_mon"] + 0.25 * df["delta_ens3_lag1"], df["delta_ens3_lag1"])
        df["delta_qblend_mon75"] = wf_quantile_asymmetric_blend(df, df["delta_hybrid_mon75"].to_numpy(float), "pred_q65_mon", "pred_q35_mon")
        anc_hybrid = np.where(mon_daytime, 0.75 * df["lag7"] + 0.25 * df["lag1"], df["lag1"])
        df["delta_full_v12"] = wf_hourly_shrink(df, df["delta_qblend_mon75"].to_numpy(float), anc_hybrid)

    # --- v3 5構造アンサンブル ---
    df["v3_cal_ens5"] = (
        0.15 * df["pred_v3_cal_day"] +
        0.25 * df["pred_v3_cal_wide"] +
        0.20 * df["pred_v3_cal_wide_deep"] +
        0.15 * df["pred_v3_cal_blk4"] +
        0.25 * df["pred_v3_cal_blk6"]
    )
    df["v3_cal_corr"] = wf_anchor_dilution_correction(df, df["v3_cal_ens5"].to_numpy(float), cm)
    df["v3_qblend_corr"] = wf_quantile_asymmetric_blend(df, df["v3_cal_corr"].to_numpy(float), "pred_v3_cal_q65", "pred_v3_cal_q35")
    anc_cal = df["anc_cal"].to_numpy(float)
    df["v3_full_corr"] = wf_hourly_shrink(df, df["v3_qblend_corr"].to_numpy(float), anc_cal)

    if "delta_full_v12" in df.columns:
        df["v3_v2_super_ens"] = 0.80 * df["v3_full_corr"] + 0.20 * df["delta_full_v12"]

    # --- v4 Sub-2.00 Super-Ensemble ---
    if "anc_sim" in df.columns and "d1_sol_d" in df.columns:
        v4_res = build_v4_sub2_predictions(df, cm)
        for k_col, arr in v4_res.items():
            df[k_col] = arr

    candidates = [
        c for c in [
            "lag1",
            "anc_cal",
            "anc_sim",
            "pred_day_lag1",
            "delta_full_v12",
            "v3_full_corr",
            "v3_v2_super_ens",
            "v4_prior_q_base",
            "v4_guard_shield",
            "v4_causal_ewma",
            "v4_sub2_super",
        ] if c in df.columns
    ]

    fold_rows = []
    for k, g in df.groupby("fold_idx"):
        y_f = g["y"].to_numpy(float)
        r = {"fold": g["fold"].iloc[0], "n": len(g)}
        for c in candidates:
            r[c] = float(np.mean(np.abs(y_f - g[c].to_numpy(float))))
        fold_rows.append(r)

    f_df = pd.DataFrame(fold_rows)
    w = f_df["n"].to_numpy(float)
    prod_col = "pred_day_lag1" if "pred_day_lag1" in f_df.columns else "lag1"
    prod_mae = float(np.average(f_df[prod_col], weights=w))

    s = pd.DataFrame({
        "weighted_mae": {c: float(np.average(f_df[c], weights=w)) for c in candidates},
        "worst_fold": {c: float(f_df[c].max()) for c in candidates},
        "last_fold": {c: float(f_df[c].iloc[-1]) for c in candidates},
        "vs_prod_diff": {c: float(np.average(f_df[c], weights=w) - prod_mae) for c in candidates},
    }).sort_values("weighted_mae")

    f_df.to_csv(CACHE_DIR / "fast_tune_v5_fold_metrics.csv", index=False, encoding="utf-8-sig")
    s.to_csv(CACHE_DIR / "fast_tune_v5_summary.csv", encoding="utf-8-sig")

    el = time.time() - t0
    print(f"=== 高速チューニング v6 (v4 Sub-2.00 対応) 完了 (計算時間: {el:.2f}秒) ===")
    print("\n[全6 fold ウォークフォワード検証結果]")
    print(s.round(4).to_string())

    print("\n[Fold別詳細 (上位モデル vs 従来本番 pred_day_lag1)]")
    top4 = [c for c in s.index if c not in ("lag1", "anc_cal", "anc_sim")][:5]
    base_show = [c for c in ["fold", "n", "pred_day_lag1", "delta_full_v12", "v3_v2_super_ens"] if c in f_df.columns]
    show_cols = base_show + [c for c in top4 if c not in base_show]
    print(f_df[show_cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
