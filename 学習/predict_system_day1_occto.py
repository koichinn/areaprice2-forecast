#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""システムDAY1予測 v4 (OCCTO入り・v4 Sub-2.00 Super-Ensemble:
 カレンダー同期アンカー + 予報類似度アンカー(anc_sim) + JEPX入札逼迫特徴量 + 偽平均回帰シールド + 因果的Dual EWMA残差フィードバック)。
 replay : 評価の最終fold(30日)を同じ手順(リークゼロ)で再現し、精度(MAE)を検証
 predict: 最新データで学習し、DAY1の48コマを予測(OCCTOが無い日はOCCTOなしモデルに自動切替)
"""
from __future__ import annotations
import argparse, json, re, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import jpholiday
from sklearn.ensemble import HistGradientBoostingRegressor
import evaluate_system_day1_levers as ev

DB = Path(r"P:\エリアプライス2\database\統合データ")
SRC = DB / "training_v8_price_regime" / "jepx_training_v8_price_regime_system_natwf_day1.parquet"
EV = Path(r"P:\エリアプライス2\学習")
EV_IN = EV / "evaluation_system_day1_levers_from_chain" / "system_natwf_day1_without_occto.parquet"
DA8 = EV / "evaluation_system_day1_levers_da" / "occto_da8.parquet"
EV_PRED = EV / "evaluation_system_day1_levers_da_noact" / "levers_best_predictions.csv"
FUT_W = DB / "future_weather_features_system_day1.parquet"
FUT_P = DB / "future_price_features_system_day1.parquet"
W_ARCH = DB / "weather_archive"
DL = Path(r"P:\エリアプライス2\database\occtoデータ\downloads")
OUT = Path(r"P:\エリアプライス2\予測")
CALIB_JSON = Path(__file__).resolve().parent / "system_day1_v3_calib.json"

P = "pre1100_occto_"
OCC5 = [P + c for c in ("block_demand_mw", "block_supply_mw", "block_reserve_mw", "wide_reserve_rate_pct", "wide_utilization_pct")]
NOACT = r"^(?!システムプライス|area_price_yen_kwh|売り入札量|買い入札量|約定総量|pr_).*_lag\d+d$"
EXCLUDE_FEATURES = {"solar_curtailment_lag1d"}  # 昼帯過剰下落(谷外れ)防止

DAY = ev.DAY
REST = [h for h in range(24) if h not in DAY]
WIDE = list(range(7, 20))
BLOCKS4 = [[21, 22, 23, 0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16], [17, 18, 19, 20]]
BLOCKS6 = [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22, 23]]
DAY_SPLIT = [DAY, REST]
WIDE_SPLIT = [WIDE, [h for h in range(24) if h not in WIDE]]

DELTA_BASE_COLS = [
    "pre1100_occto_wide_utilization_pct",
    "pre1100_occto_wide_reserve_rate_pct",
    "pre1100_occto_block_demand_mw",
    "pre1100_occto_block_reserve_mw",
    "pre1100_occto_block_supply_mw",
    "wf_nat_solar_effective_wm2_mean",
    "wf_nat_direct_radiation_wm2_mean",
    "wf_nat_shortwave_radiation_wm2_mean",
    "wf_nat_cloud_cover_pct_mean",
    "wf_nat_wind_power_proxy_mean",
    "wf_nat_wind_speed_10m_mean",
    "wf_nat_temperature_c_mean",
    "wf_nat_temperature_c_std",
    "wf_nat_humidity_pct_mean",
    "wf_nat_precipitation_mm_30min_mean",
    "wf_tokyo_solar_effective_wm2",
    "wf_kansai_solar_effective_wm2",
    "wf_chubu_solar_effective_wm2",
    "wf_kyushu_solar_effective_wm2",
    "wf_tohoku_solar_effective_wm2",
    "wf_chugoku_solar_effective_wm2",
    "wf_shikoku_solar_effective_wm2",
    "wf_shortwave_radiation_wm2_day_sum",
    "wf_direct_radiation_wm2_day_sum",
    "wf_cloud_cover_pct_day_mean",
    "wf_cdd_24",
    "wf_hdd_12",
]


def load_training(src):
    d = pd.read_parquet(src)
    d = d[[c for c in d.columns if "occto" not in c.lower() and not c.startswith("pr_interact_")]]
    d[ev.TCOL] = pd.to_datetime(d[ev.TCOL])
    d = d[pd.to_numeric(d[ev.TARGET], errors="coerce").notna()].sort_values(ev.TCOL).reset_index(drop=True)
    ex = pd.read_parquet(DA8)
    ex[ev.TCOL] = pd.to_datetime(ex[ev.TCOL])
    ex = ex.groupby(ev.TCOL, as_index=False)[OCC5].mean()
    n0 = len(d)
    d = d.merge(ex, on=ev.TCOL, how="left", validate="many_to_one")
    assert len(d) == n0
    lag1 = ev.find_lag1(list(d.columns))
    d = d[d[lag1].notna()].reset_index(drop=True)
    feats_all, dropped = ev.pick_features(d, lag1)
    return d, lag1, [c for c in feats_all if not re.search(NOACT, c) and c not in EXCLUDE_FEATURES]


def get_calendar_masks(dts: pd.Series) -> dict[str, np.ndarray]:
    dates = dts.dt.date
    dow = dts.dt.dayofweek.to_numpy()
    hour = dts.dt.hour.to_numpy()
    min_d = pd.Timestamp(dates.min()) - pd.Timedelta(days=25)
    max_d = pd.Timestamp(dates.max()) + pd.Timedelta(days=5)
    all_d = pd.date_range(min_d, max_d, freq="D").date
    hol_set = {
        x for x in all_d
        if jpholiday.is_holiday(x)
        or (x.month == 8 and x.day in (13, 14, 15))
        or (x.month == 1 and x.day in (2, 3))
        or (x.month == 12 and x.day in (29, 30, 31))
    }
    hol_set_obon = {
        x for x in all_d
        if jpholiday.is_holiday(x)
        or (x.month == 8 and x.day in (10, 11, 12, 13, 14, 15, 16))
        or (x.month == 1 and x.day in (2, 3))
        or (x.month == 12 and x.day in (29, 30, 31))
    }

    def is_rest(x):
        w = pd.Timestamp(x).dayofweek
        return (w >= 5) or (x in hol_set)

    def is_rest_obon(x):
        w = pd.Timestamp(x).dayofweek
        return (w >= 5) or (x in hol_set_obon)

    is_hol = np.array([x in hol_set for x in dates], dtype=bool)
    is_rst = np.array([is_rest(x) for x in dates], dtype=bool)
    rst_obon = np.array([is_rest_obon(x) for x in dates], dtype=bool)
    prev1_hol = np.array([(pd.Timestamp(x) - pd.Timedelta(days=1)).date() in hol_set for x in dates], dtype=bool)
    prev2_hol = np.array([(pd.Timestamp(x) - pd.Timedelta(days=2)).date() in hol_set for x in dates], dtype=bool)
    prev7_hol = np.array([(pd.Timestamp(x) - pd.Timedelta(days=7)).date() in hol_set for x in dates], dtype=bool)
    prev1_rest = np.array([is_rest((pd.Timestamp(x) - pd.Timedelta(days=1)).date()) for x in dates], dtype=bool)
    prev2_rest = np.array([is_rest((pd.Timestamp(x) - pd.Timedelta(days=2)).date()) for x in dates], dtype=bool)
    rst_yesterday = np.array([is_rest_obon((pd.Timestamp(x) - pd.Timedelta(days=1)).date()) for x in dates], dtype=bool)
    rst_lag2 = np.array([is_rest_obon((pd.Timestamp(x) - pd.Timedelta(days=2)).date()) for x in dates], dtype=bool)
    rst_lag7 = np.array([is_rest_obon((pd.Timestamp(x) - pd.Timedelta(days=7)).date()) for x in dates], dtype=bool)
    rst_lag14 = np.array([is_rest_obon((pd.Timestamp(x) - pd.Timedelta(days=14)).date()) for x in dates], dtype=bool)
    rst_tomorrow = np.array([is_rest_obon((pd.Timestamp(x) + pd.Timedelta(days=1)).date()) for x in dates], dtype=bool)
    daytime = (hour >= 7) & (hour <= 21)

    return {
        "dow": dow,
        "hour": hour,
        "is_hol": is_hol,
        "is_rst": is_rst,
        "rst_obon": rst_obon,
        "prev1_hol": prev1_hol,
        "prev2_hol": prev2_hol,
        "prev7_hol": prev7_hol,
        "prev1_rest": prev1_rest,
        "prev2_rest": prev2_rest,
        "rst_yesterday": rst_yesterday,
        "rst_lag2": rst_lag2,
        "rst_lag7": rst_lag7,
        "rst_lag14": rst_lag14,
        "rst_tomorrow": rst_tomorrow,
        "daytime": daytime,
        "m_mon_p7hol": (~is_hol) & (dow == 0) & prev7_hol & daytime,
        "m_mon_norm": (~is_hol) & (dow == 0) & (~prev7_hol) & daytime,
        "m_tue_ahol": (~is_hol) & (dow == 1) & prev1_hol & daytime,
        "m_hol_wk": is_hol & (dow >= 1) & (~prev1_rest) & prev2_rest,
    }


def engineer_v3_features(d: pd.DataFrame, lag1_col: str):
    """リークゼロでカレンダー同期アンカー(anc_cal)・予報類似度アンカー(anc_sim)・安定レジームアンカー(anc_regime)とv3/v4特徴量群を生成"""
    d = d.sort_values(ev.TCOL).reset_index(drop=True)
    t = pd.to_datetime(d[ev.TCOL])
    cm = get_calendar_masks(t)
    dow, hour = cm["dow"], cm["hour"]
    is_hol, is_rst, rst_obon = cm["is_hol"], cm["is_rst"], cm["rst_obon"]
    prev1_hol, prev2_hol, prev7_hol = cm["prev1_hol"], cm["prev2_hol"], cm["prev7_hol"]
    prev1_rest, prev2_rest, daytime = cm["prev1_rest"], cm["prev2_rest"], cm["daytime"]

    l1 = np.nan_to_num(pd.to_numeric(d[lag1_col], errors="coerce").to_numpy(float), nan=0.0)
    l2 = np.nan_to_num(pd.to_numeric(d["システムプライス(円/kWh)_lag2d"], errors="coerce").to_numpy(float), nan=l1)
    l7 = np.nan_to_num(pd.to_numeric(d["システムプライス(円/kWh)_lag7d"], errors="coerce").to_numpy(float), nan=l1)
    l14 = np.nan_to_num(pd.to_numeric(d["システムプライス(円/kWh)_lag14d"], errors="coerce").to_numpy(float), nan=l7)

    w1 = np.ones(len(d), dtype=float)
    w2 = np.zeros(len(d), dtype=float)
    w7 = np.zeros(len(d), dtype=float)
    w14 = np.zeros(len(d), dtype=float)

    m_hol_mon = is_hol & (dow == 0)
    w1[m_hol_mon], w7[m_hol_mon] = 0.55, 0.45

    m_hol_wk = is_hol & (dow >= 1) & (~prev1_rest) & prev2_rest
    w1[m_hol_wk], w2[m_hol_wk] = 0.25, 0.75

    m_hol_mid = is_hol & (dow >= 1) & (~prev1_rest) & (~prev2_rest)
    w1[m_hol_mid], w7[m_hol_mid] = 0.50, 0.50

    m_mon_p7hol = (~is_hol) & (dow == 0) & prev7_hol & daytime
    w1[m_mon_p7hol], w14[m_mon_p7hol] = 0.25, 0.75

    m_mon_norm = (~is_hol) & (dow == 0) & (~prev7_hol) & daytime
    w1[m_mon_norm], w7[m_mon_norm] = 0.25, 0.75

    m_tue_ahol = (~is_hol) & (dow == 1) & prev1_hol & daytime
    w1[m_tue_ahol], w7[m_tue_ahol] = 0.45, 0.55

    m_wed_ahol = (~is_hol) & (dow >= 2) & (dow <= 4) & prev1_hol & daytime
    w1[m_wed_ahol], w2[m_wed_ahol], w7[m_wed_ahol] = 0.40, 0.25, 0.35

    m_sat = (~is_hol) & (dow == 5) & daytime
    w1[m_sat], w7[m_sat], w14[m_sat] = 0.55, 0.25, 0.20

    anc_cal = w1 * l1 + w2 * l2 + w7 * l7 + w14 * l14

    lag_stack = np.vstack([l1, l2, l7, l14])
    sorted_lags = np.sort(lag_stack, axis=0)
    rest_base = 0.5 * (sorted_lags[0] + sorted_lags[1])
    wkday_base = np.nanmedian(np.vstack([anc_cal, l7, l14]), axis=0)
    anc_regime = np.where(is_rst, rest_base, wkday_base)

    # 予報類似度アンカー (anc_sim) の構築 (前日10時前の気象・OCCTO予報のみ使用)
    sol_s = pd.to_numeric(d.get("wf_nat_solar_effective_wm2_mean", pd.Series(0.0, index=d.index)), errors="coerce").fillna(0.0)
    util_s = pd.to_numeric(d.get("pre1100_occto_wide_utilization_pct", pd.Series(82.0, index=d.index)), errors="coerce").fillna(82.0)
    s_df = pd.DataFrame({"sol": sol_s.to_numpy(float), "util": util_s.to_numpy(float)}, index=t)
    day_mask = (hour >= 8) & (hour <= 16)
    daily_stats = s_df[day_mask].groupby(s_df[day_mask].index.normalize()).mean()

    d_norm = t.dt.normalize()
    sol_d = d_norm.map(daily_stats["sol"]).to_numpy(float)
    util_d = d_norm.map(daily_stats["util"]).to_numpy(float)
    sol_l1 = (d_norm - pd.Timedelta(days=1)).map(daily_stats["sol"]).to_numpy(float)
    util_l1 = (d_norm - pd.Timedelta(days=1)).map(daily_stats["util"]).to_numpy(float)
    sol_l7 = (d_norm - pd.Timedelta(days=7)).map(daily_stats["sol"]).to_numpy(float)
    util_l7 = (d_norm - pd.Timedelta(days=7)).map(daily_stats["util"]).to_numpy(float)
    sol_l14 = (d_norm - pd.Timedelta(days=14)).map(daily_stats["sol"]).to_numpy(float)
    util_l14 = (d_norm - pd.Timedelta(days=14)).map(daily_stats["util"]).to_numpy(float)

    dist_7 = np.nan_to_num(np.abs(sol_d - sol_l7) / 50.0 + np.abs(util_d - util_l7) / 3.0, nan=1.0) + np.where(rst_obon == cm["rst_lag7"], 0.0, 2.5)
    dist_14 = np.nan_to_num(np.abs(sol_d - sol_l14) / 50.0 + np.abs(util_d - util_l14) / 3.0, nan=1.0) + np.where(rst_obon == cm["rst_lag14"], 0.0, 2.5)
    dist_1 = np.nan_to_num(np.abs(sol_d - sol_l1) / 50.0 + np.abs(util_d - util_l1) / 3.0, nan=1.0) + np.where(rst_obon == cm["rst_yesterday"], 0.0, 2.0)

    temp = 1.5
    anc_sim = anc_cal.copy()
    ratio_14_vs_7 = np.exp(-dist_14 / temp) / (np.exp(-dist_7 / temp) + np.exp(-dist_14 / temp))
    best_wknd_lag = (1.0 - ratio_14_vs_7) * l7 + ratio_14_vs_7 * l14
    dist_wknd = np.minimum(dist_7, dist_14)

    m_sat_sim = (dow == 5) & daytime & (~cm["rst_yesterday"])
    w_l1_sat = np.clip(0.55 * np.exp(-(dist_1 - dist_wknd) / (2.0 * temp)), 0.20, 0.80)
    anc_sim[m_sat_sim] = w_l1_sat[m_sat_sim] * l1[m_sat_sim] + (1.0 - w_l1_sat[m_sat_sim]) * best_wknd_lag[m_sat_sim]

    m_mon_sim = (dow == 0) & (~rst_obon) & daytime
    anc_sim[m_mon_sim] = 0.25 * l1[m_mon_sim] + 0.75 * best_wknd_lag[m_mon_sim]

    m_obon_mon = (dow == 0) & rst_obon & daytime
    anc_sim[m_obon_mon] = 0.70 * l1[m_obon_mon] + 0.30 * l7[m_obon_mon]

    cm["anc_sim"] = anc_sim
    cm["sol_d"] = np.nan_to_num(sol_d, nan=0.0)
    cm["util_d"] = np.nan_to_num(util_d, nan=82.0)
    cm["d1_sol_d"] = np.nan_to_num(sol_d - sol_l1, nan=0.0)
    cm["d1_util_d"] = np.nan_to_num(util_d - util_l1, nan=0.0)

    res_rate = pd.to_numeric(d.get("pre1100_occto_wide_reserve_rate_pct", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    util_rate = pd.to_numeric(d.get("pre1100_occto_wide_utilization_pct", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    dem_mw = pd.to_numeric(d.get("pre1100_occto_block_demand_mw", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    sol_eff = pd.to_numeric(d.get("wf_nat_solar_effective_wm2_mean", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)

    pre_cols = {}
    pre_cols["nl_occ_res_tight12"] = np.maximum(0.0, 12.0 - np.nan_to_num(res_rate, nan=15.0)) ** 1.5
    pre_cols["nl_occ_res_tight9"] = np.maximum(0.0, 9.0 - np.nan_to_num(res_rate, nan=15.0)) ** 2.0
    pre_cols["nl_occ_util_high88"] = np.maximum(0.0, np.nan_to_num(util_rate, nan=80.0) - 88.0) ** 1.5
    pre_cols["nl_occ_util_high91"] = np.maximum(0.0, np.nan_to_num(util_rate, nan=80.0) - 91.0) ** 2.0
    pre_cols["nl_solar_surplus_zone"] = np.maximum(0.0, np.nan_to_num(sol_eff, nan=0.0) - 200.0) * np.maximum(0.0, 83.0 - np.nan_to_num(util_rate, nan=85.0))
    pre_cols["nl_net_demand_proxy"] = np.nan_to_num(dem_mw, nan=40000.0) - 40.0 * np.nan_to_num(sol_eff, nan=0.0)

    s_sol = pd.Series(sol_eff)
    s_res = pd.Series(res_rate)
    s_util = pd.Series(util_rate)
    s_net = pd.Series(pre_cols["nl_net_demand_proxy"])

    pre_cols["ramp1h_solar"] = (s_sol - s_sol.shift(2)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_solar"] = (s_sol - s_sol.shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_fwd_solar"] = (s_sol.shift(-4) - s_sol).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_occ_res"] = (s_res - s_res.shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_occ_util"] = (s_util - s_util.shift(4)).fillna(0.0).to_numpy(float)
    pre_cols["ramp2h_net_demand"] = (s_net - s_net.shift(4)).fillna(0.0).to_numpy(float)

    d = pd.concat([d, pd.DataFrame(pre_cols, index=d.index)], axis=1)

    use_cols = [c for c in DELTA_BASE_COLS + list(pre_cols.keys()) if c in d.columns]
    base_df = d.set_index(ev.TCOL)[use_cols]

    curr_v = base_df.to_numpy(float)
    v1 = base_df.reindex(t - pd.Timedelta(days=1)).to_numpy(float)
    v2 = base_df.reindex(t - pd.Timedelta(days=2)).to_numpy(float)
    v7 = base_df.reindex(t - pd.Timedelta(days=7)).to_numpy(float)
    v14 = base_df.reindex(t - pd.Timedelta(days=14)).to_numpy(float)

    d1_m = curr_v - v1
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
    for i, c in enumerate(use_cols):
        new_cols[f"d1_{c}"] = d1_m[:, i]
        new_cols[f"d7_{c}"] = d7_m[:, i]
        new_cols[f"dcal_{c}"] = dcal_m[:, i]
        if c.startswith("pre1100_occto_") or c.startswith("nl_") or "solar_effective" in c or "shortwave" in c:
            new_cols[f"d2_{c}"] = d2_m[:, i]
            new_cols[f"d14_{c}"] = d14_m[:, i]

    med_lag = np.nanmedian(lag_stack, axis=0)
    min_lag = np.nanmin(lag_stack, axis=0)
    new_cols["anc_cal_level"] = anc_cal
    new_cols["anc_regime_level"] = anc_regime
    new_cols["anc_cal_vs_regime"] = anc_cal - anc_regime
    new_cols["anc_cal_vs_med_lag"] = anc_cal - med_lag
    new_cols["anc_cal_vs_min_lag"] = anc_cal - min_lag
    new_cols["anc_cal_vs_lag7"] = anc_cal - l7
    new_cols["anc_cal_vs_lag14"] = anc_cal - l14
    new_cols["lag1_vs_med_lag"] = l1 - med_lag
    new_cols["lag1_vs_lag7"] = l1 - l7
    new_cols["lag1_vs_lag2"] = l1 - l2

    new_cols["cal_is_hol"] = is_hol.astype(float)
    new_cols["cal_is_rest"] = is_rst.astype(float)
    new_cols["cal_prev1_hol"] = prev1_hol.astype(float)
    new_cols["cal_prev2_hol"] = prev2_hol.astype(float)
    new_cols["cal_prev7_hol"] = prev7_hol.astype(float)
    new_cols["cal_prev1_rest"] = prev1_rest.astype(float)
    new_cols["cal_prev2_rest"] = prev2_rest.astype(float)
    new_cols["cal_dow"] = dow.astype(float)

    if "dcal_pre1100_occto_wide_utilization_pct" in new_cols:
        new_cols["ix_anc_dev_x_dcal_util"] = (anc_cal - med_lag) * new_cols["dcal_pre1100_occto_wide_utilization_pct"]
        new_cols["ix_lag1_dev_x_d1_util"] = (l1 - med_lag) * new_cols["d1_pre1100_occto_wide_utilization_pct"]
    if "dcal_wf_nat_solar_effective_wm2_mean" in new_cols:
        new_cols["ix_anc_dev_x_dcal_solar"] = (anc_cal - med_lag) * new_cols["dcal_wf_nat_solar_effective_wm2_mean"]
        new_cols["ix_anc_level_x_dcal_solar"] = anc_cal * new_cols["dcal_wf_nat_solar_effective_wm2_mean"]

    # v4 特徴量: JEPX 入札逼迫特徴量 & クリーン気象平均回帰特徴量
    sell_1 = pd.to_numeric(d.get("売り入札量(kWh)_lag1d", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    buy_1 = pd.to_numeric(d.get("買い入札量(kWh)_lag1d", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    sell_7 = pd.to_numeric(d.get("売り入札量(kWh)_lag7d", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    buy_7 = pd.to_numeric(d.get("買い入札量(kWh)_lag7d", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    sell_14 = pd.to_numeric(d.get("売り入札量(kWh)_lag14d", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)
    buy_14 = pd.to_numeric(d.get("買い入札量(kWh)_lag14d", pd.Series(np.nan, index=d.index)), errors="coerce").to_numpy(float)

    imb_1 = (buy_1 - sell_1) / np.maximum(1.0, sell_1)
    imb_7 = (buy_7 - sell_7) / np.maximum(1.0, sell_7)
    imb_14 = (buy_14 - sell_14) / np.maximum(1.0, sell_14)

    new_cols["bid_imb_ratio_lag1"] = np.nan_to_num(imb_1, nan=0.0)
    new_cols["bid_imb_diff_1_7"] = np.nan_to_num(imb_1 - imb_7, nan=0.0)
    new_cols["bid_imb_diff_1_14"] = np.nan_to_num(imb_1 - imb_14, nan=0.0)
    new_cols["sell_drop_ratio_1_7"] = np.nan_to_num((sell_1 - sell_7) / np.maximum(1.0, sell_7), nan=0.0)
    new_cols["buy_surge_ratio_1_7"] = np.nan_to_num((buy_1 - buy_7) / np.maximum(1.0, buy_7), nan=0.0)
    new_cols["anc_sim_diff_cal"] = anc_sim - anc_cal
    new_cols["anc_sim_level"] = anc_sim
    new_cols["is_tomorrow_rest"] = cm["rst_tomorrow"].astype(float)

    d1_sol_slot = np.nan_to_num(new_cols.get("d1_wf_nat_solar_effective_wm2_mean", np.zeros(len(d))), nan=0.0)
    d1_util_slot = np.nan_to_num(new_cols.get("d1_pre1100_occto_wide_utilization_pct", np.zeros(len(d))), nan=0.0)
    weather_relief = np.maximum(0.0, d1_sol_slot / 60.0) + np.maximum(0.0, -d1_util_slot / 3.0)
    weather_stress = np.maximum(0.0, -d1_sol_slot / 60.0) + np.maximum(0.0, d1_util_slot / 3.0)
    new_cols["cond_mr_down_on_relief"] = np.maximum(0.0, anc_sim - anc_regime) * weather_relief
    new_cols["cond_persist_up_on_stress"] = np.maximum(0.0, anc_sim - anc_regime) * weather_stress

    cm["bid_imb_diff_1_7"] = new_cols["bid_imb_diff_1_7"]

    d_out = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)
    return d_out, anc_cal, anc_regime, cm


def mk_v3_model(loss="absolute_error", quantile=None, iter_scale=0.55, seed=0, leaf=31, lr=0.045, it=450, msl=35):
    kw = dict(loss=loss, max_iter=max(10, int(it * iter_scale)), learning_rate=lr,
              max_leaf_nodes=leaf, min_samples_leaf=msl, l2_regularization=1.2,
              early_stopping=False, random_state=seed)
    if loss == "quantile" and quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def blocks_fit_pred(Xtr, htr, rtr, Xte, hte, blocks, loss="absolute_error", quantile=None, iter_scale=0.55, leaf=31, lr=0.045, it=450, msl=35, seed=0):
    p = np.zeros(len(Xte))
    used = np.zeros(len(p), bool)
    for hs in blocks:
        m = np.isin(htr, hs)
        s = np.isin(hte, hs)
        if s.any() and m.sum() > 400:
            mdl = mk_v3_model(loss=loss, quantile=quantile, iter_scale=iter_scale, seed=seed, leaf=leaf, lr=lr, it=it, msl=msl)
            mdl.fit(Xtr[m], rtr[m])
            p[s] = mdl.predict(Xte[s])
            used |= s
    if not used.all():
        mdl = mk_v3_model(loss=loss, quantile=quantile, iter_scale=iter_scale, seed=seed, leaf=leaf, lr=lr, it=it, msl=msl)
        mdl.fit(Xtr, rtr)
        p[~used] = mdl.predict(Xte[~used])
    return p


def load_calib_config() -> dict:
    if CALIB_JSON.exists():
        return json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    return {}


def apply_dilution_corr(base_arr: np.ndarray, p_lag7: np.ndarray, p_reg: np.ndarray, l1: np.ndarray, l2: np.ndarray, l7: np.ndarray, l14: np.ndarray, m1: np.ndarray, m2: np.ndarray, m3: np.ndarray, m4: np.ndarray, gamma: float) -> np.ndarray:
    delta_vec = np.zeros(len(base_arr), dtype=float)
    delta_vec[m1] = np.maximum(0.0, l14[m1] - l1[m1])
    delta_vec[m2] = 0.50 * np.maximum(0.0, l7[m2] - l1[m2]) + 0.50 * (p_lag7[m2] - base_arr[m2])
    delta_vec[m3] = 0.50 * np.maximum(0.0, l7[m3] - l1[m3]) + 0.50 * (p_lag7[m3] - base_arr[m3])
    delta_vec[m4] = 0.50 * (l2[m4] - l1[m4]) + 0.50 * (p_reg[m4] - base_arr[m4])
    return base_arr + gamma * delta_vec


def v3_ensemble_predict(d_panel: pd.DataFrame, anc_cal: np.ndarray, anc_regime: np.ndarray, cm: dict, tr_mask: np.ndarray, te_mask: np.ndarray, cols: list[str], calib: dict, iter_scale: float = 0.55) -> np.ndarray:
    """v4 Sub-2.00 Super-Ensemble:
    v3 カレンダー同期アンサンブル + v4 予報類似度アンカー(anc_sim)・JEPX入札逼迫モデル + v2 デルタモデル
    + 偽平均回帰シールド + 需給・境界レジーム較正 + 因果的 Dual EWMA 残差フィードバック
    """
    t_all = pd.to_datetime(d_panel[ev.TCOL])
    dates_norm = t_all.dt.normalize()
    te_min_date = dates_norm[te_mask].min()
    # 評価対象期間 + 直前21日分の確定実績日(Causal EWMA算出用)をまとめて推論対象とする
    ctx_tr_mask = tr_mask & (dates_norm >= (te_min_date - pd.Timedelta(days=22))) & (dates_norm < te_min_date)
    eval_mask = ctx_tr_mask | te_mask

    X = d_panel[cols].to_numpy(dtype=np.float32)
    cfg_all = load_calib_config()
    v4_cols = [c for c in cfg_all.get("v4_feats", cols) if c in d_panel.columns and (c in cols or "occto" not in c.lower())]
    X_v4 = d_panel[v4_cols].to_numpy(dtype=np.float32)

    y = pd.to_numeric(d_panel[ev.TARGET], errors="coerce").to_numpy(float)
    l1 = pd.to_numeric(d_panel["システムプライス(円/kWh)_lag1d"], errors="coerce").to_numpy(float)
    l2 = pd.to_numeric(d_panel["システムプライス(円/kWh)_lag2d"], errors="coerce").fillna(pd.Series(l1)).to_numpy(float)
    l7 = pd.to_numeric(d_panel["システムプライス(円/kWh)_lag7d"], errors="coerce").fillna(pd.Series(l1)).to_numpy(float)
    l14 = pd.to_numeric(d_panel["システムプライス(円/kWh)_lag14d"], errors="coerce").fillna(pd.Series(l7)).to_numpy(float)
    anc_sim = cm["anc_sim"]
    hour = cm["hour"]
    dow = cm["dow"]
    is_rest = cm["is_rst"]

    res_cal = y - anc_cal
    res_sim = y - anc_sim
    res_lag1 = y - l1
    res_lag7 = y - l7
    res_reg = y - anc_regime

    Xtr, htr = X[tr_mask], hour[tr_mask]
    Xev, hev = X[eval_mask], hour[eval_mask]
    Xtr_v4, Xev_v4 = X_v4[tr_mask], X_v4[eval_mask]

    anc_cal_ev = anc_cal[eval_mask]
    anc_sim_ev = anc_sim[eval_mask]
    anc_reg_ev = anc_regime[eval_mask]
    l1_ev, l2_ev, l7_ev, l14_ev = l1[eval_mask], l2[eval_mask], l7[eval_mask], l14[eval_mask]

    # 1) v3 カレンダー同期 5構造 + q65
    p_cal_day = anc_cal_ev + blocks_fit_pred(Xtr, htr, res_cal[tr_mask], Xev, hev, DAY_SPLIT, iter_scale=iter_scale, seed=0)
    p_cal_wide = anc_cal_ev + blocks_fit_pred(Xtr, htr, res_cal[tr_mask], Xev, hev, WIDE_SPLIT, iter_scale=iter_scale, seed=1)
    p_cal_wide_deep = anc_cal_ev + blocks_fit_pred(Xtr, htr, res_cal[tr_mask], Xev, hev, WIDE_SPLIT, iter_scale=iter_scale, leaf=47, lr=0.035, it=520, msl=28, seed=2)
    p_cal_blk4 = anc_cal_ev + blocks_fit_pred(Xtr, htr, res_cal[tr_mask], Xev, hev, BLOCKS4, iter_scale=iter_scale, seed=3)
    p_cal_blk6 = anc_cal_ev + blocks_fit_pred(Xtr, htr, res_cal[tr_mask], Xev, hev, BLOCKS6, iter_scale=iter_scale, seed=4)
    p_cal_q65 = anc_cal_ev + blocks_fit_pred(Xtr, htr, res_cal[tr_mask], Xev, hev, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=5)

    p_lag1_wide = l1_ev + blocks_fit_pred(Xtr, htr, res_lag1[tr_mask], Xev, hev, WIDE_SPLIT, iter_scale=iter_scale, seed=7)
    p_lag1_blk6 = l1_ev + blocks_fit_pred(Xtr, htr, res_lag1[tr_mask], Xev, hev, BLOCKS6, iter_scale=iter_scale, seed=8)
    p_lag7_wide = l7_ev + blocks_fit_pred(Xtr, htr, res_lag7[tr_mask], Xev, hev, WIDE_SPLIT, iter_scale=iter_scale, seed=9)
    p_reg_wide = anc_reg_ev + blocks_fit_pred(Xtr, htr, res_reg[tr_mask], Xev, hev, WIDE_SPLIT, iter_scale=iter_scale, seed=10)

    # 2) v4 予報類似度アンカー (anc_sim) & JEPX 入札逼迫特徴量モデル (L1 + q65)
    p_v4_l1 = anc_sim_ev + blocks_fit_pred(Xtr_v4, htr, res_sim[tr_mask], Xev_v4, hev, WIDE_SPLIT, loss="absolute_error", iter_scale=0.45, seed=42)
    p_v4_q65 = anc_sim_ev + blocks_fit_pred(Xtr_v4, htr, res_sim[tr_mask], Xev_v4, hev, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=0.45, seed=43)

    ens5 = 0.15 * p_cal_day + 0.25 * p_cal_wide + 0.20 * p_cal_wide_deep + 0.15 * p_cal_blk4 + 0.25 * p_cal_blk6

    # 3) スケール不変アンカー希釈補正
    gamma = float(calib.get("gamma", 0.275))
    m1 = cm["m_mon_p7hol"][eval_mask]
    m2 = cm["m_mon_norm"][eval_mask]
    m3 = cm["m_tue_ahol"][eval_mask]
    m4 = cm["m_hol_wk"][eval_mask]

    v3_cal_corr = apply_dilution_corr(ens5, p_lag7_wide, p_reg_wide, l1_ev, l2_ev, l7_ev, l14_ev, m1, m2, m3, m4, gamma)
    q65_corr = apply_dilution_corr(p_cal_q65, p_lag7_wide, p_reg_wide, l1_ev, l2_ev, l7_ev, l14_ev, m1, m2, m3, m4, gamma)
    v4_l1_corr = apply_dilution_corr(p_v4_l1, p_lag7_wide, p_reg_wide, l1_ev, l2_ev, l7_ev, l14_ev, m1, m2, m3, m4, gamma)
    v4_q65_corr = apply_dilution_corr(p_v4_q65, p_lag7_wide, p_reg_wide, l1_ev, l2_ev, l7_ev, l14_ev, m1, m2, m3, m4, gamma)

    w_v4 = 0.20
    l1_mix = (1.0 - w_v4) * v3_cal_corr + w_v4 * v4_l1_corr
    q65_mix = (1.0 - w_v4) * q65_corr + w_v4 * v4_q65_corr
    anc_mix = (1.0 - w_v4) * anc_cal_ev + w_v4 * anc_sim_ev

    dow_ev = dow[eval_mask]
    rst_ev = is_rest[eval_mask]
    is_peak = (hev >= 15) & (hev <= 21)
    is_midday = (hev >= 10) & (hev <= 14)
    w_q = np.where(rst_ev, 0.36, np.where(is_peak, 0.78, np.where(is_midday, 0.52, 0.55)))
    base_q = (1.0 - w_q) * l1_mix + w_q * q65_mix

    alpha_map = calib.get("alpha_v4_by_hour", calib.get("alpha_by_hour", {}))
    shrunk_q = base_q.copy()
    res_q = base_q - anc_mix
    for h in range(24):
        mh = (hev == h)
        if mh.any():
            alpha = float(alpha_map.get(str(h), 1.0))
            shrunk_q[mh] = anc_mix[mh] + alpha * res_q[mh]

    # v2 月曜ハイブリッドデルタアンサンブル
    lag1_ens2 = 0.55 * p_lag1_wide + 0.45 * p_lag1_blk6
    mon_day_ev = (dow_ev == 0) & (hev >= 7) & (hev <= 21)
    delta_hybrid = np.where(mon_day_ev, 0.75 * p_lag7_wide + 0.25 * lag1_ens2, lag1_ens2)
    p_base = 0.82 * shrunk_q + 0.18 * delta_hybrid

    # 4) 予報類似度アンカー補正 + 偽平均回帰シールド (False Mean-Reversion Shield)
    p_sim = p_base + (0.25 * (1.0 - w_v4)) * (anc_sim_ev - anc_cal_ev)
    d1_sol_ev = cm["d1_sol_d"][eval_mask]
    d1_util_ev = cm["d1_util_d"][eval_mask]
    sol_cur_ev = cm["sol_d"][eval_mask]
    util_cur_ev = cm["util_d"][eval_mask]

    high_regime_persist = ((anc_sim_ev - anc_reg_ev) > 2.0) & (d1_sol_ev <= 20.0) & (d1_util_ev >= -1.5)
    res_sim_p = p_sim - anc_sim_ev
    p_guard = np.where(high_regime_persist & (res_sim_p < 0), anc_sim_ev + 0.55 * res_sim_p, p_sim)

    offsets = calib.get("offsets_v4", {})
    tight_surge = (((util_cur_ev >= 88.0) & (d1_util_ev >= 3.5) & (sol_cur_ev <= 220.0)) | ((util_cur_ev >= 90.0) & (d1_sol_ev <= -60.0) & (d1_util_ev >= 2.5))) & (hev >= 8) & (hev <= 21)
    ext_sol_drop = (d1_sol_ev <= -180.0) & (l1_ev >= 24.0) & (hev >= 8) & (hev <= 21)

    p_g3 = p_guard.copy()
    p_g3[tight_surge] += float(offsets.get("tight_surge", 1.25))
    p_g3[ext_sol_drop] += float(offsets.get("ext_sol_drop", -1.70))

    # 5) 厳密因果的 Dual EWMA 残差フィードバック (d_p < d_cur の確定実績のみ参照)
    # 学習データ内に含まれる直前21日 (ctx_tr_mask) は in-sample 残差が過小になるため 1.65 倍で OOF スケールに補正
    dates_ev = dates_norm[eval_mask].reset_index(drop=True)
    y_ev = y[eval_mask]
    is_ctx_ev = ctx_tr_mask[eval_mask]
    raw_err_ev = np.where(is_ctx_ev, 1.65 * (y_ev - p_g3), y_ev - p_g3)
    blk_id = np.zeros(len(Xev), dtype=int)
    for bi, hs in enumerate(BLOCKS6):
        blk_id[np.isin(hev, hs)] = bi

    df_tmp = pd.DataFrame({"date": dates_ev, "blk": blk_id, "err": raw_err_ev, "is_rest": rst_ev})
    valid_err = df_tmp[df_tmp["err"].notna()]
    blk_daily = valid_err.groupby(["date", "blk"])["err"].median().unstack("blk")
    date_is_rest = df_tmp.groupby("date")["is_rest"].first()
    avail_past_dates = sorted(blk_daily.index.unique())

    ewma_all_arr = np.zeros(len(Xev), dtype=float)
    ewma_same_arr = np.zeros(len(Xev), dtype=float)
    for d_cur in sorted(dates_ev.unique()):
        m_cur = (dates_ev == d_cur).to_numpy()
        cur_rst = bool(date_is_rest.loc[d_cur])
        past_dates = [d_p for d_p in avail_past_dates if d_p < d_cur and (d_cur - d_p).days <= 21]
        if not past_dates:
            continue
        ages = np.array([(d_cur - d_p).days for d_p in past_dates], dtype=float)
        w_all = 0.5 ** ((ages - 1.0) / 6.0)
        w_all /= w_all.sum()
        same_mask = np.array([bool(date_is_rest.loc[d_p]) == cur_rst for d_p in past_dates], dtype=bool)
        for bi in range(len(BLOCKS6)):
            mb = m_cur & (blk_id == bi)
            if not mb.any() or bi not in blk_daily.columns:
                continue
            vals = np.nan_to_num(np.array([blk_daily.loc[d_p, bi] for d_p in past_dates], dtype=float), nan=0.0)
            ewma_all_arr[mb] = float(np.dot(w_all, vals))
            if same_mask.any():
                w_s = 0.5 ** ((ages[same_mask] - 1.0) / 6.0)
                w_s /= w_s.sum()
                ewma_same_arr[mb] = float(np.dot(w_s, vals[same_mask]))
            else:
                ewma_same_arr[mb] = ewma_all_arr[mb]

    p_ewma = p_g3 + 0.20 * np.clip(ewma_all_arr, -3.5, 3.5) + 0.15 * np.clip(ewma_same_arr, -3.5, 3.5)

    # 6) Walk-Forward 需給・境界レジーム較正 (前日10時前の確定特徴量のみ)
    imb_diff_ev = cm["bid_imb_diff_1_7"][eval_mask]
    rst_yest_ev = cm["rst_yesterday"][eval_mask]
    rst_tom_ev = cm["rst_tomorrow"][eval_mask]

    m_scarcity = (~rst_ev) & (imb_diff_ev >= 0.08) & (d1_util_ev >= 0.0) & (hev >= 8) & (hev <= 21)
    m_abundance = (imb_diff_ev <= -0.08) & (d1_sol_ev >= 0.0) & (d1_util_ev <= 0.0) & (hev >= 8) & (hev <= 21)
    m_mon_surge = (dow_ev == 0) & (~rst_ev) & (d1_util_ev >= 6.0) & (hev >= 7) & (hev <= 21)
    m_sun_eve = rst_ev & (~rst_tom_ev) & (hev >= 16) & (hev <= 23)
    m_mon_morn = (~rst_ev) & rst_yest_ev & (hev >= 0) & (hev <= 7)
    m_sat_morn = rst_ev & (~rst_yest_ev) & (hev >= 0) & (hev <= 7)
    m_tue_tight = (dow_ev == 1) & (~rst_ev) & (util_cur_ev >= 86.0) & (d1_util_ev >= 1.0) & (hev >= 8) & (hev <= 21)

    p_final = p_ewma.copy()
    p_final[m_scarcity] += float(offsets.get("scarcity", 0.68))
    p_final[m_abundance] += float(offsets.get("abundance", -0.22))
    p_final[m_mon_surge] += float(offsets.get("mon_surge", 0.75))
    p_final[m_sun_eve] += float(offsets.get("sun_eve", 0.70))
    p_final[m_mon_morn] += float(offsets.get("mon_morn", -0.40))
    p_final[m_sat_morn] += float(offsets.get("sat_morn", 0.45))
    p_final[m_tue_tight] += float(offsets.get("tue_tight", 0.50))

    te_in_eval = te_mask[eval_mask]
    return p_final[te_in_eval]


def replay(a):
    cfg = load_calib_config()
    iter_scale = a.iter_scale if a.iter_scale is not None else float(cfg.get("iter_scale_default", 0.55))
    d, lag1, cols_base = load_training(a.src or SRC)
    l7_col = "システムプライス(円/kWh)_lag7d"
    l14_col = "システムプライス(円/kWh)_lag14d"
    d = d[d[lag1].notna() & d[l7_col].notna() & d[l14_col].notna()].reset_index(drop=True)
    d_v3, anc_cal, anc_regime, cm = engineer_v3_features(d, lag1)

    v3_feats = [c for c in cfg.get("v3_feats", cols_base) if c in d_v3.columns]
    t = d_v3[ev.TCOL]
    y = d_v3[ev.TARGET].to_numpy(float)

    # OOFキャッシュが存在する場合はその最終fold期間に合わせて比較可能にする
    cache_p = EV / "fast_eval_cache" / "oof_cache_system_day1.parquet"
    if cache_p.exists() and a.date is None:
        oof_df = pd.read_parquet(cache_p)
        f5_dts = pd.to_datetime(oof_df.loc[oof_df["fold_idx"] == 5, "target_datetime"])
        s0 = f5_dts.min().normalize()
        end = f5_dts.max().normalize() + pd.Timedelta(days=1)
    else:
        end = t.max().normalize() + pd.Timedelta(days=1)
        s0 = end - pd.Timedelta(days=a.fold_days)

    te = ((t >= s0) & (t < end)).to_numpy()
    tr = (t < s0 - pd.Timedelta(days=1)).to_numpy()

    print(f"学習行={int(tr.sum()):,} テスト行={int(te.sum()):,} 特徴量={len(v3_feats)} (OCCTO {sum(c in OCC5 for c in v3_feats)}本) テスト期間 {s0.date()}~{(end - pd.Timedelta(days=1)).date()}")
    pred = v3_ensemble_predict(d_v3, anc_cal, anc_regime, cm, tr, te, v3_feats, cfg.get("calib_fold5", {}), iter_scale=iter_scale)
    mine = pd.DataFrame({"dt": t[te].to_numpy(), "new": pred, "y": y[te]})
    mae_new = float((mine.new - mine.y).abs().mean())
    print(f"v4_sub2_super 最終fold MAE = {mae_new:.4f}")

    if EV_PRED.exists():
        old = pd.read_csv(EV_PRED, encoding="utf-8-sig")
        old["dt"] = pd.to_datetime(old["dt"])
        m = mine.merge(old[["dt", "pred"]], on="dt", how="inner")
        if not m.empty:
            df = (m.new - m.pred).abs()
            nd = int((df > 1e-4).sum())
            mae_old = float((m.pred - m.y).abs().mean())
            print(f"評価の保存値と突合できた行={len(m):,}/{len(mine):,}  違う行={nd:,}/{len(m):,}  最大差={df.max():.4g}  MAE: 今回={float((m.new - m.y).abs().mean()):.4f} 旧評価={mae_old:.4f}")
    return 0


def day1_date(now):
    now = pd.Timestamp(now)
    return now.normalize() + pd.Timedelta(days=1 if now.hour < 11 else 2)


AR9 = ["北海道", "東北", "東京", "中部", "北陸", "関西", "中国", "四国", "九州"]
V = {"広域ブロック需要(MW)": "block_demand_mw", "広域ブロック供給力(MW)": "block_supply_mw", "広域ブロック予備力(MW)": "block_reserve_mw",
     "広域予備率(%)": "wide_reserve_rate_pct", "広域使用率(%)": "wide_utilization_pct"}


def occto_daily(day, idx):
    p = DL / f"occto_wide_block_reserve_dayaftertomorrow_{day:%Y%m%d}.csv"
    if not p.exists():
        return None, f"{p.name} がありません"
    for enc in ("utf-8-sig", "cp932"):
        try:
            d = pd.read_csv(p, encoding=enc)
            break
        except UnicodeDecodeError:
            pass
    d.columns = [str(c).strip().rstrip(".") for c in d.columns]
    t = d["時刻"].astype(str).str.replace(r"^(\d):", r"0\1:", regex=True)
    d["dt"] = pd.to_datetime(d["対象年月日"].astype(str).str.replace("-", "/")) + pd.to_timedelta(pd.to_numeric(t.str[:2]) * 60 + pd.to_numeric(t.str[3:5]) - 30, unit="m")
    x = pd.concat([d[d[a].astype(str).str.contains("〇")][["dt"] + list(V)] for a in AR9]).groupby("dt")[list(V)].mean()
    x = x.rename(columns={k: P + v for k, v in V.items()}).reindex(idx)
    x.index.name = ev.TCOL
    if int(x.notna().all(axis=1).sum()) < 48:
        return None, f"{p.name} は48コマ揃っていません"
    return x.reset_index(), ""


def build_recent_context_rows(day: pd.Timestamp, d_tr: pd.DataFrame) -> pd.DataFrame:
    """d_tr の最終行と予測対象日 day の間にある直近15日分の気象予報(wf_)・OCCTO(OCC5)をアーカイブから補完する"""
    ctx_start = day - pd.Timedelta(days=15)
    ctx_end = day
    ctx_idx = pd.date_range(ctx_start, ctx_end - pd.Timedelta(minutes=30), freq="30min")
    ctx = pd.DataFrame({ev.TCOL: ctx_idx})

    wf_frames = []
    if W_ARCH.exists():
        for fp in sorted(W_ARCH.glob("future_wf_*.parquet"))[-15:]:
            try:
                w = pd.read_parquet(fp)
                w[ev.TCOL] = pd.to_datetime(w[ev.TCOL])
                wf_frames.append(w)
            except Exception:
                pass
    if FUT_W.exists():
        w = pd.read_parquet(FUT_W)
        w[ev.TCOL] = pd.to_datetime(w[ev.TCOL])
        wf_frames.append(w)
    if wf_frames:
        wf_all = pd.concat(wf_frames, ignore_index=True).drop_duplicates(ev.TCOL, keep="last")
        wf_cols = [c for c in wf_all.columns if c == ev.TCOL or c.startswith("wf_")]
        ctx = ctx.merge(wf_all[wf_cols], on=ev.TCOL, how="left")

    occ_frames = []
    if DA8.exists():
        ex = pd.read_parquet(DA8)
        ex[ev.TCOL] = pd.to_datetime(ex[ev.TCOL])
        occ_frames.append(ex.groupby(ev.TCOL, as_index=False)[OCC5].mean())
    for k in range(15, 0, -1):
        d_k = day - pd.Timedelta(days=k)
        idx_k = pd.date_range(d_k, periods=48, freq="30min")
        o_k, _ = occto_daily(d_k, idx_k)
        if o_k is not None:
            occ_frames.append(o_k)
    if occ_frames:
        occ_all = pd.concat(occ_frames, ignore_index=True).drop_duplicates(ev.TCOL, keep="last")
        ctx = ctx.merge(occ_all[[ev.TCOL] + OCC5], on=ev.TCOL, how="left")

    existing_t = set(pd.to_datetime(d_tr[ev.TCOL]))
    return ctx[~ctx[ev.TCOL].isin(existing_t)].reset_index(drop=True)


def predict(a):
    cfg = load_calib_config()
    iter_scale = a.iter_scale if a.iter_scale is not None else float(cfg.get("iter_scale_default", 0.55))
    d, lag1, feats_occ = load_training(a.src or SRC)
    day = pd.Timestamp(a.date) if a.date else day1_date(datetime.now())
    idx = pd.date_range(day, periods=48, freq="30min")

    X = pd.DataFrame({ev.TCOL: idx})
    for f in (FUT_W, FUT_P):
        x = pd.read_parquet(f)
        x[ev.TCOL] = pd.to_datetime(x[ev.TCOL])
        X = X.merge(x.drop_duplicates(ev.TCOL), on=ev.TCOL, how="left")
    X["forecast_horizon_days"] = 1

    occ, why = occto_daily(day, idx)
    use = occ is not None
    if use:
        X = X.merge(occ, on=ev.TCOL, how="left")
    else:
        print(f"OCCTOなし: {why} → OCCTOなしのv4モデルで予測します")

    if X[lag1].isna().any():
        print(f"ERROR {lag1} が欠損です。JEPXマスターに前日の価格が入っていません")
        return 1

    ctx_missing = build_recent_context_rows(day, d)
    panel = pd.concat([d, ctx_missing, X], ignore_index=True).drop_duplicates(ev.TCOL, keep="last").sort_values(ev.TCOL).reset_index(drop=True)
    panel_v3, anc_cal, anc_regime, cm = engineer_v3_features(panel, lag1)

    v3_feats_cfg = cfg.get("v3_feats", feats_occ)
    if use:
        cols = [c for c in v3_feats_cfg if c in panel_v3.columns]
    else:
        cols = [c for c in v3_feats_cfg if c in panel_v3.columns and "occto" not in c.lower() and "occ_" not in c.lower()]

    l7_col = "システムプライス(円/kWh)_lag7d"
    l14_col = "システムプライス(円/kWh)_lag14d"
    tr_mask = (
        panel_v3[ev.TCOL].isin(d[ev.TCOL])
        & pd.to_numeric(panel_v3[ev.TARGET], errors="coerce").notna()
        & panel_v3[lag1].notna()
        & panel_v3[l7_col].notna()
        & panel_v3[l14_col].notna()
    ).to_numpy(bool)
    te_mask = panel_v3[ev.TCOL].isin(idx).to_numpy(bool)

    if int(te_mask.sum()) != 48:
        print(f"ERROR 予測対象コマ数が48ではありません ({int(te_mask.sum())})")
        return 1

    nn = panel_v3.loc[te_mask, cols].isna().sum()
    print(f"DAY1={day.date()} 特徴量={len(cols)}列 欠損のある列={int((nn > 0).sum())}" + (f" {nn[nn > 0].head(5).to_dict()}" if (nn > 0).any() else ""))

    t_tr_max = panel_v3.loc[tr_mask, ev.TCOL].max()
    print(f"学習行={int(tr_mask.sum()):,} 最終target={t_tr_max} v4_sub2_super アンサンブル学習中(iter_scale={iter_scale})...")
    pred = v3_ensemble_predict(panel_v3, anc_cal, anc_regime, cm, tr_mask, te_mask, cols, cfg.get("calib_prod", {}), iter_scale=iter_scale)

    now = datetime.now().astimezone().isoformat()
    mdl = "v4_sub2_super+OCCTO5" if use else "v4_sub2_super(OCCTOなし)"
    anc_l1_te = pd.to_numeric(panel_v3.loc[te_mask, lag1], errors="coerce").to_numpy(float)
    o = pd.DataFrame({
        "target_datetime": panel_v3.loc[te_mask, ev.TCOL].to_numpy(),
        "predicted_system_price_yen_kwh": pred,
        "anchor_lag1d": anc_l1_te,
        "model_used": mdl,
        "forecast_created_at": now,
        "train_rows": int(tr_mask.sum()),
        "train_last_target": str(t_tr_max),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    o.to_csv(OUT / "forecast_system_day1_occto_latest.csv", index=False, encoding="utf-8-sig")
    h = OUT / "forecast_system_day1_occto_history.csv"
    o.to_csv(h, mode="a", header=not h.exists(), index=False, encoding="utf-8" if h.exists() else "utf-8-sig")
    print(f"予測: 平均={o.predicted_system_price_yen_kwh.mean():.2f} 最小={o.predicted_system_price_yen_kwh.min():.2f} 最大={o.predicted_system_price_yen_kwh.max():.2f} モデル={mdl}")
    try:
        f = pd.read_csv(OUT / "forecast_day1_to_day7_latest.csv", encoding="utf-8-sig")
        f["target_datetime"] = pd.to_datetime(f["target_datetime"])
        f = f[(f["area"] == "system") & f["target_datetime"].isin(idx)].set_index("target_datetime")["predicted_price_yen_kwh"]
        c = o.set_index("target_datetime")["predicted_system_price_yen_kwh"].reindex(f.index)
        print(f"既存の予測(直接予測)との平均差={float((c - f).mean()):.2f} 平均絶対差={float((c - f).abs().mean()):.2f} ({len(f)}コマ)")
    except Exception as e:
        print("既存の予測との比較はできません:", str(e)[:80])
    print("出力:", OUT / "forecast_system_day1_occto_latest.csv")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["predict", "replay"], default="predict")
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--fold-days", type=int, default=30)
    ap.add_argument("--iter-scale", type=float, default=None)
    a = ap.parse_args()
    if a.iter_scale is not None:
        ev.ITER_SCALE = a.iter_scale
    sys.exit(replay(a) if a.mode == "replay" else predict(a))


if __name__ == "__main__":
    main()
