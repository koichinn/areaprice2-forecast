#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""東京エリア価格 DAY1 専用予測スクリプト v4 (Grand Ensemble: HGB + LightGBM Physical Blocks)
- Path A (Tokyo Direct): 東京カレンダー同期 (anc_cal) + 予報類似度 (anc_sim) + L1 (DAY/WIDE/DEEP/BLK4/BLK6) + q65/q35
- Path B (Reconstructed): System v4_sub2_super + 東京プレミアム (anc_cal_prem / anc_sim_prem / WIDE / BLOCKS6 / q65)
- Path C (Partial-Coupling Z07): 0.70 * System v4_sub2_super + Z07 (WIDE / BLOCKS6 / q65)
- LightGBM Physical Blocks: 5物理ブロック（夜間、朝ランプ、太陽光&FC閉塞、夕点灯&火力ピーク、深夜減衰）× 分位点q60/q65/q70/L1
- グランドアンサンブル: 昼帯(07:00-20:30) HGB 70% + LGB 30%、夜間帯 HGB 90% + LGB 10%
出力: DAY1 48コマの予測値 (predicted_price_yen_kwh)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, r"P:\エリアプライス2\学習")
import build_fast_eval_cache_tokyo_day1 as B
import fast_tune_tokyo_day1 as ft

INTEG = Path(r"P:\エリアプライス2\database\統合データ")
OUT_DIR = Path(r"P:\エリアプライス2\予測")
CALIB_JSON = Path(r"P:\エリアプライス2\学習\tokyo_day1_v4_calib.json")

BLOCKS_PHYSICAL = [
    [0, 1, 2, 3, 4, 5, 6],       # Night baseline
    [7, 8],                      # Morning ramp
    [9, 10, 11, 12, 13, 14],     # Solar peak & FC saturation
    [15, 16, 17, 18, 19],        # Evening lighting ramp & thermal peak
    [20, 21, 22, 23],            # Late night decay
]


def lgb_fit_predict(Xtr, htr, rtr, Xte, hte, blocks, objective="mae", alpha=0.5, n_est=180, lr=0.045, leaf=31, seed=42):
    p = np.zeros(len(Xte))
    valid_tr = np.isfinite(rtr)
    for hs in blocks:
        m = np.isin(htr, hs) & valid_tr
        s = np.isin(hte, hs)
        if s.any() and m.sum() > 400:
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
                verbose=-1
            )
            mdl.fit(Xtr[m], rtr[m])
            p[s] = mdl.predict(Xte[s])
    return p


def load_calib() -> tuple[float, float, list[str] | None, list[str] | None, float, float]:
    w_lgb_day = 0.30
    w_lgb_night = 0.10
    if CALIB_JSON.exists():
        try:
            cfg = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
            wq = float(cfg.get("wq", ft.DEFAULT_WQ))
            wa = float(cfg.get("wa", ft.DEFAULT_WA))
            tok_f = cfg.get("tok_feats")
            prem_f = cfg.get("prem_feats")
            ge = cfg.get("grand_ensemble", {})
            w_lgb_day = float(ge.get("w_lgb_day", 0.30))
            w_lgb_night = float(ge.get("w_lgb_night", 0.10))
            return wq, wa, tok_f, prem_f, w_lgb_day, w_lgb_night
        except Exception:
            pass
    return ft.DEFAULT_WQ, ft.DEFAULT_WA, None, None, w_lgb_day, w_lgb_night


# ---- 水準ドリフト補正 (Level Drift Correction) ----
PRED_HISTORY_CSV = OUT_DIR / "tokyo_day1_v4_pred_history.csv"
DRIFT_K = 0.50          # 補正の強さ
DRIFT_SPAN = 3          # 日次残差 EWMA の span
DRIFT_LOOKBACK_DAYS = 14  # これより古い残差しか無い場合は補正しない


def load_pred_history() -> pd.DataFrame:
    """P: 側の履歴と、GitHub Actions 上ではリポジトリにコミット済みの履歴を統合して返す。"""
    paths = [PRED_HISTORY_CSV]
    ws = os.environ.get("GITHUB_WORKSPACE")
    if ws:
        paths.append(Path(ws) / "予測" / PRED_HISTORY_CSV.name)
    frames = []
    for p in paths:
        if p.exists():
            try:
                h = pd.read_csv(p, encoding="utf-8-sig")
                h["target_datetime"] = pd.to_datetime(h["target_datetime"])
                frames.append(h)
            except Exception as e:
                print(f"  [Drift Warning] 履歴読み込み失敗 {p}: {e}")
    if not frames:
        return pd.DataFrame(columns=["target_datetime", "pred_raw", "source"])
    h = pd.concat(frames, ignore_index=True)
    h["_live"] = (h.get("source", "") == "live").astype(int)
    h = h.sort_values(["target_datetime", "_live"]).drop_duplicates("target_datetime", keep="last")
    return h.drop(columns="_live").reset_index(drop=True)


def compute_drift_bias(actual: pd.DataFrame, day: pd.Timestamp) -> float:
    """確定済み実績(D-1以前)と補正前予測の日次中央値残差の EWMA を返す。"""
    h = load_pred_history()
    if h.empty:
        return 0.0
    m = h.merge(actual, on="target_datetime", how="inner")
    m = m[m["target_datetime"] < day.normalize()]
    m = m[m["target_datetime"] >= day.normalize() - pd.Timedelta(days=DRIFT_LOOKBACK_DAYS)]
    if m.empty:
        return 0.0
    m["date"] = m["target_datetime"].dt.normalize()
    daily = (m["y"] - m["pred_raw"]).groupby(m["date"]).median().sort_index()
    if len(daily) < 2:
        return 0.0
    return float(daily.ewm(span=DRIFT_SPAN).mean().iloc[-1])


def update_pred_history(idx: pd.DatetimeIndex, pred_raw: np.ndarray) -> None:
    h = load_pred_history()
    h = h[~h["target_datetime"].isin(idx)]
    new = pd.DataFrame({"target_datetime": idx, "pred_raw": pred_raw, "source": "live"})
    h = pd.concat([h, new], ignore_index=True).sort_values("target_datetime")
    h.to_csv(PRED_HISTORY_CSV, index=False, encoding="utf-8-sig")


def predict_tokyo_day1(day: pd.Timestamp, p_sys: np.ndarray, iter_scale: float = 0.35) -> pd.DataFrame:
    idx = pd.date_range(day, periods=48, freq="30min")
    assert len(p_sys) == 48, f"p_sys must have 48 slots, got {len(p_sys)}"

    # 将来DAY1の特徴量（future_area_day1_area.parquet の東京の行）を extra として読み込む
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
                print(f"  将来DAY1東京特徴量を future_area_day1_area.parquet から読み込みました ({len(extra_df)}行)")
        except Exception as e:
            print(f"  future_area_day1_area.parquet 読み込みスキップ: {e}")

    # 1. データと特徴量エンジニアリング
    print(f"=== 東京DAY1予測 (Grand Ensemble v4) 開始: {day.strftime('%Y-%m-%d')} ===")
    t_start = time.time()
    d, aux, cand_cols, must_keep = B.load_and_engineer_tokyo(extra=extra_df)

    t = pd.to_datetime(d["target_datetime"])
    te = (t >= idx[0]) & (t <= idx[-1])
    assert te.sum() == 48, f"テストコマ数が48ではありません: {te.sum()}"
    tr = t < idx[0]

    # 1.1 スプレッド＆価格水準ドリフト特徴量エンジニアリング (確定D-1以前の価格から生成)
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
    print(f"  スプレッド＆水準ドリフト特徴量追加完了: {len(new_drift_feats)}列")

    hour = t.dt.hour.to_numpy()
    htr, hte = hour[tr], hour[te]
    y_tok = d["target"].to_numpy(float)
    y_sys = d["y_sys"].to_numpy(float)
    y_prem = d["y_prem"].to_numpy(float)
    y_z07 = y_tok - 0.70 * y_sys

    anc_cal_tok = aux["anc_cal_tok"]
    anc_sim_tok = aux["anc_sim_tok"]
    anc_reg_tok = aux["anc_regime_tok"]
    anc_cal_prem = aux["anc_cal_prem"]
    anc_sim_prem = aux["anc_sim_prem"]
    anc_cal_z07 = aux["anc_cal_z07"]
    anc_sim_z07 = aux["anc_sim_z07"]

    res_tok_cal = y_tok - anc_cal_tok
    res_tok_sim = y_tok - anc_sim_tok
    res_tok_lag1 = y_tok - d["area_price_yen_kwh_lag1d"].to_numpy(float)
    res_tok_lag7 = y_tok - d["area_price_yen_kwh_lag7d"].to_numpy(float)
    res_prem_cal = y_prem - anc_cal_prem
    res_prem_sim = y_prem - anc_sim_prem
    res_z07_cal = y_z07 - anc_cal_z07
    res_z07_sim = y_z07 - anc_sim_z07

    # 特徴量抽出 (calib.jsonに保存されていればキャッシュを使用)
    wq, wa, tok_feats_saved, prem_feats_saved, w_lgb_day, w_lgb_night = load_calib()
    if tok_feats_saved and all(c in d.columns for c in tok_feats_saved):
        tok_feats = tok_feats_saved
    else:
        print("  tok_feats を再計算中...")
        tok_feats = B.select_top_features(d, cand_cols, res_tok_cal, must_keep, top_k=195)

    # スプレッドドリフト特徴量を確実に含める
    tok_feats = sorted(list(set(tok_feats) | set(new_drift_feats)))

    if prem_feats_saved and all(c in d.columns for c in prem_feats_saved):
        prem_feats = prem_feats_saved
    else:
        print("  prem_feats を再計算中...")
        prem_feats = B.select_top_features(d, cand_cols, res_prem_cal, must_keep, top_k=195)

    print(f"  特徴量選択完了: tok_feats={len(tok_feats)}, prem_feats={len(prem_feats)}")
    sys.stdout.flush()

    X_tok = d[tok_feats].to_numpy(dtype=np.float32)
    X_prem = d[prem_feats].to_numpy(dtype=np.float32)

    DAY_SPLIT = [B.DAY, [h for h in range(24) if h not in B.DAY]]
    WIDE_SPLIT = [B.WIDE, [h for h in range(24) if h not in B.WIDE]]

    # 3. HGB モデル推論
    c = {}
    t0 = time.time()
    print("  [1/4] Path A (Tokyo Direct HGB) 推論中...")
    sys.stdout.flush()
    c["pred_tok_cal_day"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, DAY_SPLIT, iter_scale=iter_scale, seed=10)
    c["pred_tok_cal_wide"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=11)
    c["pred_tok_cal_wide_deep"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, leaf=47, lr=0.035, it=520, msl=28, seed=12)
    c["pred_tok_cal_blk4"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, B.BLOCKS4, iter_scale=iter_scale, seed=13)
    c["pred_tok_cal_blk6"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=14)
    c["pred_tok_cal_q65"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=15)
    c["pred_tok_sim_wide"] = anc_sim_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=17)
    c["pred_tok_sim_blk6"] = anc_sim_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=18)
    c["pred_tok_sim_q65"] = anc_sim_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=19)
    c["pred_tok_lag1_wide"] = d["area_price_yen_kwh_lag1d"].to_numpy(float)[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_lag1[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=20)
    c["pred_tok_lag1_blk6"] = d["area_price_yen_kwh_lag1d"].to_numpy(float)[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_lag1[tr], X_tok[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=21)

    print(f"  [2/4] Path B (Reconstructed Premium HGB) 推論中... ({time.time() - t0:.1f}s)")
    sys.stdout.flush()
    c["pred_prem_cal_wide"] = anc_cal_prem[te] + B.blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=30)
    c["pred_prem_cal_blk6"] = anc_cal_prem[te] + B.blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=31)
    c["pred_prem_cal_q65"] = anc_cal_prem[te] + B.blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=32)

    print(f"  [3/4] Path C (Partial Coupling Z07 HGB) 推論中... ({time.time() - t0:.1f}s)")
    sys.stdout.flush()
    c["pred_z07_cal_wide"] = anc_cal_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=40)
    c["pred_z07_cal_blk6"] = anc_cal_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=41)
    c["pred_z07_cal_q65"] = anc_cal_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=42)
    c["pred_z07_sim_wide"] = anc_sim_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=43)
    c["pred_z07_sim_blk6"] = anc_sim_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=44)
    c["pred_z07_sim_q65"] = anc_sim_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=45)

    # アンカー配列
    c["anc_sim"] = anc_sim_tok[te]
    c["anc_sim_z07"] = anc_sim_z07[te]
    c["anc_cal"] = anc_cal_tok[te]
    c["anc_cal_sys"] = aux["anc_cal_sys"][te]

    # HGB スーパーアンサンブル合成
    comp = ft.components(c, p_sys)
    v4_hgb = ft.final_pred(comp, wq, wa)

    # 4. LightGBM 物理ブロック推論
    print(f"  [4/4] LightGBM 物理ブロックモデル推論中... ({time.time() - t0:.1f}s)")
    sys.stdout.flush()
    lgb_tok_q60 = anc_cal_tok[te] + lgb_fit_predict(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, BLOCKS_PHYSICAL, objective="quantile", alpha=0.60, seed=101)
    lgb_tok_q65 = anc_cal_tok[te] + lgb_fit_predict(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, BLOCKS_PHYSICAL, objective="quantile", alpha=0.65, seed=102)
    lgb_tok_sim_q65 = anc_sim_tok[te] + lgb_fit_predict(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, BLOCKS_PHYSICAL, objective="quantile", alpha=0.65, seed=103)
    lgb_tok_l1 = anc_cal_tok[te] + lgb_fit_predict(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, BLOCKS_PHYSICAL, objective="mae", seed=104)

    lgb_direct = 0.40 * lgb_tok_q60 + 0.30 * lgb_tok_q65 + 0.20 * lgb_tok_sim_q65 + 0.10 * lgb_tok_l1

    # 時間帯別適応ブレンド (Grand Ensemble)
    slot = np.arange(48)
    is_day = ((slot >= 14) & (slot <= 41)).astype(float)
    w_lgb = is_day * w_lgb_day + (1.0 - is_day) * w_lgb_night
    pred_final = (1.0 - w_lgb) * v4_hgb + w_lgb * lgb_direct

    # 5. 連系線物理ディカップリング (IC Physical Decoupling)
    # 東西FCの空き容量が逼迫 (<= 200MW) している昼帯コマは、システム価格連動を切り離し東京直接モデル比率を高める
    if "ic_fc_bwd_free" in d.columns:
        fc_bwd_free = pd.to_numeric(d["ic_fc_bwd_free"], errors="coerce").to_numpy(float)[te]
        fc_bwd_free = np.where(np.isnan(fc_bwd_free), 999.0, fc_bwd_free)
    else:
        fc_bwd_free = np.full(48, 999.0)

    tok_cal = 0.20 * c["pred_tok_cal_day"] + 0.25 * c["pred_tok_cal_wide"] + 0.25 * c["pred_tok_cal_wide_deep"] + 0.15 * c["pred_tok_cal_blk4"] + 0.15 * c["pred_tok_cal_blk6"]
    tok_sim = 0.55 * c["pred_tok_sim_wide"] + 0.45 * c["pred_tok_sim_blk6"]
    tok_lag1 = 0.60 * c["pred_tok_lag1_wide"] + 0.40 * c["pred_tok_lag1_blk6"]
    hgb_direct_l1 = (0.30 * tok_cal + 0.15 * tok_sim + 0.10 * tok_lag1) / 0.55
    hgb_direct_q65 = 0.60 * c["pred_tok_cal_q65"] + 0.40 * c["pred_tok_sim_q65"]
    hgb_direct = 0.70 * hgb_direct_q65 + 0.20 * hgb_direct_l1 + 0.10 * c["anc_cal"]
    pure_direct = 0.50 * hgb_direct + 0.50 * lgb_direct

    is_sat = (is_day == 1) & (fc_bwd_free <= 200.0)
    pred_final = np.where(is_sat, 0.40 * v4_hgb + 0.60 * pure_direct, pred_final)
    sat_count = int(is_sat.sum())
    if sat_count > 0:
        print(f"  [IC Decoupling] 東西FC混雑検出: {sat_count} コマで東京直接比率を強化しました")

    # 6. 水準ドリフト補正 (Level Drift Correction)
    # 確定済み (D-1以前) の「実績 - 補正前予測」の日次中央値を EWMA(span=3) で平滑化し、その 50% を加算する。
    # 厳密WF検証: MAE 2.6054 -> 2.5723。履歴は予測/tokyo_day1_v4_pred_history.csv に蓄積。
    pred_raw = pred_final.copy()
    drift_bias = 0.0
    try:
        actual = pd.DataFrame({"target_datetime": t[tr].to_numpy(), "y": y_tok[tr]}).dropna()
        drift_bias = compute_drift_bias(actual, day)
    except Exception as e:
        print(f"  [Drift Warning] ドリフト補正をスキップ: {e}")
    pred_final = pred_raw + DRIFT_K * drift_bias
    print(f"  [Drift] 水準ドリフト補正: bias={drift_bias:+.3f} 円 x k={DRIFT_K} -> {DRIFT_K * drift_bias:+.3f} 円")
    update_pred_history(idx, pred_raw)

    print(f"  全モデル推論＆グランドアンサンブル合成完了 ({time.time() - t0:.1f}s)")
    sys.stdout.flush()

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
        "anchor_lag1d": d["area_price_yen_kwh_lag1d"].to_numpy(float)[te],
        "model_used": "v4_tokyo_grand_ensemble(HGB+Enhanced_LGB_SpreadDrift+IC_Decoupling+Level_Drift)",
        "pred_raw_before_drift": pred_raw,
        "drift_adjust": DRIFT_K * drift_bias,
        "forecast_created_at": datetime.now().astimezone().isoformat(),
        "train_rows": int(tr.sum()),
    })
    return res_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="予測対象日 (YYYY-MM-DD)")
    ap.add_argument("--system", type=Path, default=OUT_DIR / "forecast_system_day1_occto_latest.csv")
    ap.add_argument("--iter-scale", type=float, default=0.35)
    a = ap.parse_args()

    sysf = pd.read_csv(a.system, encoding="utf-8-sig")
    sysf["target_datetime"] = pd.to_datetime(sysf["target_datetime"])
    day = pd.Timestamp(a.date) if a.date else sysf["target_datetime"].min().normalize()

    sub = sysf[(sysf["target_datetime"] >= day) & (sysf["target_datetime"] < day + pd.Timedelta(days=1))].copy()
    sub = sub.sort_values("target_datetime").reset_index(drop=True)
    assert len(sub) == 48, f"システム予測が48コマありません: {len(sub)}"
    sys_c = "predicted_price_yen_kwh" if "predicted_price_yen_kwh" in sub.columns else "predicted_system_price_yen_kwh"
    p_sys = sub[sys_c].to_numpy(float)

    res = predict_tokyo_day1(day, p_sys, iter_scale=a.iter_scale)

    out_csv = OUT_DIR / f"forecast_tokyo_day1_v4_{day.strftime('%Y%m%d')}.csv"
    latest_csv = OUT_DIR / "forecast_tokyo_day1_v4_latest.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    res.to_csv(latest_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 保存完了:")
    print(f"  - {out_csv}")
    print(f"  - {latest_csv}")
    print(f"\n[サマリー] 平均: {res['predicted_price_yen_kwh'].mean():.2f} 円, 最小: {res['predicted_price_yen_kwh'].min():.2f} 円, 最大: {res['predicted_price_yen_kwh'].max():.2f} 円")


if __name__ == "__main__":
    main()
