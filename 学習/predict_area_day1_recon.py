#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""エリア別 DAY1 予測 v4 (Grand Ensemble: HGB + 5物理ブロック LightGBM MAE/q65 + ドリフト適応)
- Path A (Direct): 直接予測 (HGB + 5物理ブロック LightGBM MAE + q65)
- Path B (Reconstructed): システム予測 + プレミアム予測 (HGB + 5物理ブロック LightGBM MAE + q65)
- ブレンド: 0.50 * Direct + 0.50 * Reconstructed
- 水準ドリフト補正: 直近実績誤差の EWMA (span=3, k=0.45) 適応
- 事前学習モデル (models/area_models_8area.joblib) 読み込みにより、10〜20秒の高速推論を実行
"""
from __future__ import annotations
import argparse, os, re, sys, time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, r"P:\エリアプライス2\学習")
import evaluate_area_premium_day1_clean_rolling as E
import predict_tokyo_day1_v4 as PTV4
try:
    import predict_tokyo_day1_v4_fast as PTV4F
except Exception:
    PTV4F = None

INTEG = Path(r"P:\エリアプライス2\database\統合データ")
V8DIR = INTEG / "training_v8_price_regime"
OUT = Path(r"P:\エリアプライス2\予測")
AREA_MODEL_PATH = Path(r"P:\エリアプライス2\models\area_models_8area.joblib")
JEPX_MASTER = Path(r"P:\エリアプライス2\database\JPEXデータ\data\jepx_spot_summary_master.csv")
PRED_HIST_CSV = OUT / "forecast_area_day1_recon_history.csv"

AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]
AREA_BLEND_W = {"chubu": 0.35, "chugoku": 0.80, "shikoku": 0.75, "kyushu": 0.85}
CAL_ANCHOR_AREAS = {"kansai", "chugoku"}
NOACT = r"^(?!システムプライス|area_price_yen_kwh|売り入札量|買い入札量|約定総量|pr_|premium_|dt_).*_lag\d+d$"


def compute_area_anc_cal(df: pd.DataFrame, lag1_col: str) -> tuple[np.ndarray, np.ndarray]:
    """カレンダー同期アンカー (anc_cal_area) と前日比差分を算出する。"""
    import predict_system_day1_occto as PSO
    t = pd.Series(pd.to_datetime(df.index), index=df.index)
    cm = PSO.get_calendar_masks(t)
    dow = cm["dow"]
    is_hol, prev1_hol, prev7_hol = cm["is_hol"], cm["prev1_hol"], cm["prev7_hol"]
    prev1_rest, prev2_rest, daytime = cm["prev1_rest"], cm["prev2_rest"], cm["daytime"]

    l1 = pd.to_numeric(df[lag1_col], errors="coerce").to_numpy(float)
    l2 = np.nan_to_num(pd.to_numeric(df.get("area_price_yen_kwh_lag2d", l1), errors="coerce").to_numpy(float), nan=l1)
    l7 = np.nan_to_num(pd.to_numeric(df.get("area_price_yen_kwh_lag7d", l1), errors="coerce").to_numpy(float), nan=l1)
    l14 = np.nan_to_num(pd.to_numeric(df.get("area_price_yen_kwh_lag14d", l7), errors="coerce").to_numpy(float), nan=l7)

    w1 = np.ones(len(df), dtype=float)
    w2 = np.zeros(len(df), dtype=float)
    w7 = np.zeros(len(df), dtype=float)
    w14 = np.zeros(len(df), dtype=float)

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
    return anc_cal, anc_cal - l1

BLOCKS_PHYSICAL = [
    [0, 1, 2, 3, 4, 5, 6],       # 0: Night baseline
    [7, 8],                      # 1: Morning ramp
    [9, 10, 11, 12, 13, 14],     # 2: Solar peak
    [15, 16, 17, 18, 19],        # 3: Evening lighting ramp & thermal peak
    [20, 21, 22, 23],            # 4: Late night decay
]

HP = SimpleNamespace(max_iter=400, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=24, l2=2.0, random_state=42)


def lgb_blocks_predict(block_models: dict, X: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """ブロック別 LightGBM モデル辞書から時間帯別に予測"""
    p = np.zeros(len(X), dtype=float)
    for b_idx, hs in enumerate(BLOCKS_PHYSICAL):
        s = np.isin(hours, hs)
        if s.any():
            mdl = block_models.get(b_idx)
            if hasattr(mdl, "predict"):
                p[s] = mdl.predict(X[s])
            elif isinstance(mdl, (int, float)):
                p[s] = mdl
    return p


AREA_JP = {
    "hokkaido": "北海道", "tohoku": "東北", "tokyo": "東京",
    "chubu": "中部", "hokuriku": "北陸", "kansai": "関西",
    "chugoku": "中国", "shikoku": "四国", "kyushu": "九州",
}


def load_area_pred_history() -> pd.DataFrame:
    paths = [PRED_HIST_CSV]
    ws = os.environ.get("GITHUB_WORKSPACE")
    if ws:
        paths.append(Path(ws) / "予測" / PRED_HIST_CSV.name)
    frames = []
    for p in paths:
        if p.exists():
            for enc in ("utf-8-sig", "cp932"):
                try:
                    df_h = pd.read_csv(p, encoding=enc, low_memory=False)
                    df_h["target_datetime"] = pd.to_datetime(df_h["target_datetime"], errors="coerce")
                    frames.append(df_h.dropna(subset=["target_datetime", "area"]))
                    break
                except Exception:
                    pass
    if not frames:
        return pd.DataFrame()
    h = pd.concat(frames, ignore_index=True)
    if "forecast_created_at" in h.columns:
        h = h.sort_values(["target_datetime", "forecast_created_at"])
    return h.drop_duplicates(["target_datetime", "area"], keep="last").reset_index(drop=True)


def apply_history_drift_correction(area: str, pred_prices: np.ndarray, target_dt: pd.DatetimeIndex, k=0.45, span=3) -> tuple[np.ndarray, float]:
    """確定済み(D-1以前)の補正前予測残差(実績 - pred_raw)の日次中央値 EWMA を算出し補正する"""
    if area not in AREA_JP or not JEPX_MASTER.exists():
        return pred_prices, 0.0

    try:
        hist = load_area_pred_history()
        if hist.empty:
            return pred_prices, 0.0
        hist = hist[hist["area"].astype(str).str.lower() == area].copy()
        if hist.empty:
            return pred_prices, 0.0

        jepx = None
        for enc in ("utf-8-sig", "cp932"):
            try:
                jepx = pd.read_csv(JEPX_MASTER, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                pass
        if jepx is None:
            return pred_prices, 0.0

        price_col = f"エリアプライス{AREA_JP[area]}(円/kWh)"
        if price_col not in jepx.columns:
            return pred_prices, 0.0

        jepx["dt"] = pd.to_datetime(jepx["受渡日"]) + pd.to_timedelta((pd.to_numeric(jepx["時刻コード"]) - 1) * 30, unit="m")
        act_map = pd.to_numeric(jepx.drop_duplicates("dt", keep="last").set_index("dt")[price_col], errors="coerce")

        day_norm = pd.Timestamp(target_dt.min()).normalize()
        hist = hist[(hist["target_datetime"] < day_norm) & (hist["target_datetime"] >= day_norm - pd.Timedelta(days=14))].copy()
        if hist.empty:
            return pred_prices, 0.0

        raw_s = pd.to_numeric(hist["pred_raw_before_drift"], errors="coerce") if "pred_raw_before_drift" in hist.columns else pd.Series(np.nan, index=hist.index)
        fb_s = pd.to_numeric(hist["predicted_price_yen_kwh"], errors="coerce")
        hist["pred_raw"] = raw_s.fillna(fb_s)
        hist["actual"] = hist["target_datetime"].map(act_map)
        hist = hist.dropna(subset=["pred_raw", "actual"])
        if len(hist) < 48 * 2:
            return pred_prices, 0.0

        hist["_date"] = hist["target_datetime"].dt.normalize()
        cnt = hist.groupby("_date")["actual"].count()
        valid_dates = cnt[cnt >= 24].index
        hist = hist[hist["_date"].isin(valid_dates)]
        if hist["_date"].nunique() < 2:
            return pred_prices, 0.0

        daily_med_err = (hist["actual"] - hist["pred_raw"]).groupby(hist["_date"]).median().sort_index()
        ewma_bias = float(daily_med_err.ewm(span=span, min_periods=2).mean().iloc[-1])
        if not np.isfinite(ewma_bias):
            return pred_prices, 0.0
        adj = float(np.clip(k * ewma_bias, -2.0, 2.0))
        return pred_prices + adj, adj
    except Exception:
        return pred_prices, 0.0


def prep(area, v8dir=None):
    v8dir = v8dir or V8DIR
    A, acol, _, _ = E.load(v8dir / f"jepx_training_v8_price_regime_{area}_day1.parquet", "area", area)
    P, pcol, _, _ = E.load(v8dir / f"jepx_training_v8_{area}_premium_day1.parquet", "premium", f"{area}_premium")
    ca = [c for c in E.features(A) if not re.search(NOACT, c)]
    cp = [c for c in E.features(P) if not re.search(NOACT, c)]
    try:
        import occto_area_ic_updater as OCU
        occ = OCU.load_occto_merged(include_live_downloads=False)
        if not occ.empty:
            occ_cols = OCU.select_area_occto_cols(list(occ.columns), area)
            add_a = [c for c in occ_cols if c not in ca]
            add_p = [c for c in occ_cols if c not in cp]
            if add_a:
                A = A.drop(columns=[c for c in add_a if c in A.columns], errors="ignore").join(occ[add_a], how="left")
                ca = ca + add_a
            if add_p:
                P = P.drop(columns=[c for c in add_p if c in P.columns], errors="ignore").join(occ[add_p], how="left")
                cp = cp + add_p
    except Exception as e_occ:
        print(f"WARN {area} OCCTO/IC結合スキップ: {e_occ}")
    if area in CAL_ANCHOR_AREAS:
        anc_cal, diff_l1 = compute_area_anc_cal(A, acol)
        A["anc_cal_area"] = anc_cal
        A["anc_cal_minus_l1"] = diff_l1
        for col_c in ("anc_cal_area", "anc_cal_minus_l1"):
            if col_c not in ca:
                ca.append(col_c)
    return A, acol, ca, P, pcol, cp


def fit_area(A, acol, ca, P, cp, train_end, gap_days=2):
    t = A.index.to_series().reset_index(drop=True)
    lim = train_end - pd.Timedelta(days=gap_days)
    ya = A["target"].to_numpy(float)
    anchor_col = "anc_cal_area" if "anc_cal_area" in A.columns else acol
    alag = pd.to_numeric(A[anchor_col], errors="coerce").to_numpy(float)
    yp = P.reindex(A.index)["target"].to_numpy(float)
    trb = (t < lim).to_numpy()
    tr_a = trb & ~np.isnan(alag) & ~np.isnan(ya)
    tr_p = trb & ~np.isnan(yp)
    aX = E.matrix(A, ca)
    pX = E.matrix(P.reindex(A.index), cp)
    return SimpleNamespace(aX=aX, pX=pX, ya=ya, alag=alag, yp=yp, tr_a=tr_a, tr_p=tr_p, n_a=int(tr_a.sum()), n_p=int(tr_p.sum()))


def predict_rows(F, X_a, lag_a, X_p):
    n = len(F.ya)
    Xa = np.vstack([F.aX, X_a])
    Xp = np.vstack([F.pX, X_p])
    te = np.zeros(n + len(X_a), bool)
    te[n:] = True
    ya_full = np.concatenate([F.ya - F.alag, np.zeros(len(X_a))])
    yp_full = np.concatenate([F.yp, np.zeros(len(X_p))])
    tr_a = np.concatenate([F.tr_a, np.zeros(len(X_a), bool)])
    tr_p = np.concatenate([F.tr_p, np.zeros(len(X_p), bool)])
    direct = lag_a + E.fit_predict(Xa, ya_full, tr_a, te, HP)
    prem = E.fit_predict(Xp, yp_full, tr_p, te, HP)
    return direct, prem


def blend(direct, prem, sysp, w=0.5):
    rec = sysp + prem
    return w * direct + (1 - w) * rec, rec


def predict(a):
    sysf = pd.read_csv(a.system, encoding="utf-8-sig")
    sysf["target_datetime"] = pd.to_datetime(sysf["target_datetime"])
    day = pd.Timestamp(a.date) if a.date else sysf["target_datetime"].min().normalize()
    idx = pd.date_range(day, periods=48, freq="30min")
    hours = idx.hour.to_numpy()
    is_day = (hours >= 7) & (hours <= 20)

    sp = sysf.drop_duplicates("target_datetime", keep="last").set_index("target_datetime")["predicted_system_price_yen_kwh"].reindex(idx)
    if sp.isna().any():
        print(f"ERROR 専用システム予測が{day.date()}の48コマ揃っていません(欠損{int(sp.isna().sum())})")
        return 1

    model = str(sysf["model_used"].iloc[-1])
    fa = pd.read_parquet(a.future / "future_area_day1_area.parquet")
    fp = pd.read_parquet(a.future / "future_area_day1_premium.parquet")
    for x in (fa, fp):
        x["target_datetime"] = pd.to_datetime(x["target_datetime"])

    rows, warn = [], []
    now = datetime.now().astimezone().isoformat()

    occ_live = pd.DataFrame()
    try:
        import occto_area_ic_updater as OCU
        occ_live = OCU.load_occto_merged(include_live_downloads=True)
        if not occ_live.empty:
            cov = int(occ_live.index.isin(idx).sum())
            print(f">> OCCTO/IC特徴量を読み込みました ({len(occ_live.columns)}列, DAY1カバレッジ={cov}/48コマ)")
    except Exception as e_ol:
        print(f"WARN OCCTO/ICライブ読込スキップ: {e_ol}")

    models_8 = None
    if AREA_MODEL_PATH.is_file():
        try:
            models_8 = joblib.load(AREA_MODEL_PATH)
            print(f">> 事前学習済み8エリア Grand Ensemble v4 モデルを読み込みました: {AREA_MODEL_PATH.name}")
        except Exception as e_m:
            print(f"WARN 事前学習モデル読込失敗 (オンザフライ学習へフォールバック): {e_m}")

    for ar in AREAS:
        try:
            if ar == "tokyo":
                print(">> tokyo は新アンサンブルモデル v4_tokyo_super を適用します")
                tok_done = False
                if PTV4F is not None and Path(r"P:\エリアプライス2\models\tokyo_day1_v4_models.joblib").is_file():
                    try:
                        tok_df = PTV4F.predict_tokyo_day1_fast(day, sp.to_numpy(float))
                        rows.append(tok_df)
                        print(f"OK tokyo (v4_tokyo_super fast): 予測 平均={tok_df['predicted_price_yen_kwh'].mean():.2f} 最小={tok_df['predicted_price_yen_kwh'].min():.2f} 最大={tok_df['predicted_price_yen_kwh'].max():.2f}")
                        tok_done = True
                    except Exception as ex_fast:
                        print(f"WARN tokyo fast推論失敗のため通常推論へフォールバック: {ex_fast}")
                if not tok_done:
                    try:
                        tok_df = PTV4.predict_tokyo_day1(day, sp.to_numpy(float), iter_scale=0.35)
                        rows.append(tok_df)
                        print(f"OK tokyo (v4_tokyo_super): 予測 平均={tok_df['predicted_price_yen_kwh'].mean():.2f} 最小={tok_df['predicted_price_yen_kwh'].min():.2f} 最大={tok_df['predicted_price_yen_kwh'].max():.2f}")
                    except Exception as ex_tok:
                        print(f"WARN tokyo v4_tokyo_super 失敗のため従来手法へフォールバック: {ex_tok}")
                continue

            xa = fa[fa["area"] == ar].set_index("target_datetime").reindex(idx)
            xp = fp[fp["area"] == ar].set_index("target_datetime").reindex(idx)
            if not occ_live.empty:
                occ_day = occ_live.reindex(idx)
                xa = xa.drop(columns=[c for c in occ_day.columns if c in xa.columns], errors="ignore").join(occ_day, how="left")
                xp = xp.drop(columns=[c for c in occ_day.columns if c in xp.columns], errors="ignore").join(occ_day, how="left")

            if models_8 is not None and ar in models_8:
                m = models_8[ar]
                acol = m["acol"]
                ca = m["ca"]
                cp = m["cp"]
                use_anc_cal = (m.get("anchor_type") == "anc_cal") or ("anc_cal_area" in ca)
                if use_anc_cal:
                    anc_vals, diff_l1 = compute_area_anc_cal(xa, acol)
                    xa["anc_cal_area"] = anc_vals
                    xa["anc_cal_minus_l1"] = diff_l1
                for col_need in ca:
                    if col_need not in xa.columns:
                        xa[col_need] = np.nan
                for col_need in cp:
                    if col_need not in xp.columns:
                        xp[col_need] = np.nan
                lag = xa["anc_cal_area"].to_numpy(float) if use_anc_cal else pd.to_numeric(xa[acol], errors="coerce").to_numpy(float)
                if np.isnan(lag).any():
                    raise RuntimeError(f"{acol} が欠損です(前日の{ar}の価格がJEPXマスターにありません)")
                X_a = E.matrix(xa.reset_index(), ca)
                X_p = E.matrix(xp.reset_index(), cp)

                # 1. HGB 予測
                d_hgb = lag + m["direct_model"].predict(X_a)
                pr_hgb = m["premium_model"].predict(X_p)

                # 2. Grand Ensemble v4 (LightGBM 物理ブロック MAE/q65)
                if "direct_lgb_mae" in m and "direct_lgb_q65" in m:
                    d_lgb_mae = lag + lgb_blocks_predict(m["direct_lgb_mae"], X_a, hours)
                    d_lgb_q65 = lag + lgb_blocks_predict(m["direct_lgb_q65"], X_a, hours)
                    p_lgb_mae = lgb_blocks_predict(m["premium_lgb_mae"], X_p, hours)
                    p_lgb_q65 = lgb_blocks_predict(m["premium_lgb_q65"], X_p, hours)

                    # ハイブリッド・アンサンブル (昼帯 45% HGB + 25% LGB_mae + 30% LGB_q65)
                    d = np.where(is_day, 0.45 * d_hgb + 0.25 * d_lgb_mae + 0.30 * d_lgb_q65,
                                         0.75 * d_hgb + 0.15 * d_lgb_mae + 0.10 * d_lgb_q65)
                    pr = np.where(is_day, 0.45 * pr_hgb + 0.25 * p_lgb_mae + 0.30 * p_lgb_q65,
                                          0.75 * pr_hgb + 0.15 * p_lgb_mae + 0.10 * p_lgb_q65)
                else:
                    d = d_hgb
                    pr = pr_hgb

                w_ar = float(m.get("blend_w", AREA_BLEND_W.get(ar, a.blend_weight)))
                bl_raw, rec = blend(d, pr, sp.to_numpy(float), w_ar)

                # 3. 水準ドリフト補正 (EWMA) + JEPX最低価格クリップ
                bl, drift_adj = apply_history_drift_correction(ar, bl_raw, idx, k=0.45, span=3)
                bl = np.clip(bl, 0.01, None)

                n_a = m["n_direct"]
                n_p = m["n_premium"]
            else:
                A, acol, ca, P, pcol, cp = prep(ar)
                F = fit_area(A, acol, ca, P, cp, day, a.gap_days)
                if ar in CAL_ANCHOR_AREAS:
                    anc_vals, diff_l1 = compute_area_anc_cal(xa, acol)
                    xa["anc_cal_area"] = anc_vals
                    xa["anc_cal_minus_l1"] = diff_l1
                miss = [c for c in ca if c not in xa.columns] + [c for c in cp if c not in xp.columns]
                if miss:
                    raise RuntimeError(f"将来の特徴量が足りません({len(miss)}列): {miss[:4]}")
                lag = xa["anc_cal_area"].to_numpy(float) if ar in CAL_ANCHOR_AREAS else pd.to_numeric(xa[acol], errors="coerce").to_numpy(float)
                if np.isnan(lag).any():
                    raise RuntimeError(f"{acol} が欠損です(前日の{ar}の価格がJEPXマスターにありません)")
                X_a = E.matrix(xa.reset_index(), ca)
                X_p = E.matrix(xp.reset_index(), cp)
                d, pr = predict_rows(F, X_a, lag, X_p)
                w_ar = float(AREA_BLEND_W.get(ar, a.blend_weight))
                bl_raw, rec = blend(d, pr, sp.to_numpy(float), w_ar)
                bl, drift_adj = apply_history_drift_correction(ar, bl_raw, idx, k=0.45, span=3)
                bl = np.clip(bl, 0.01, None)
                n_a = F.n_a
                n_p = F.n_p

            if np.nanmin(bl) < -5 or np.nanmax(bl) > 300:
                raise RuntimeError(f"価格が範囲外({np.nanmin(bl):.1f}〜{np.nanmax(bl):.1f})")

            rows.append(pd.DataFrame({
                "target_datetime": idx,
                "area": ar,
                "predicted_price_yen_kwh": bl,
                "direct": d,
                "reconstructed": rec,
                "system_pred": sp.to_numpy(float),
                "premium_pred": pr,
                "anchor_lag1d": lag,
                "model_used": f"grand_ens_v4(direct+sys:{model})",
                "pred_raw_before_drift": np.clip(bl_raw, 0.01, None),
                "drift_adjust": drift_adj,
                "forecast_created_at": now,
                "train_rows_direct": n_a,
                "train_rows_premium": n_p
            }))

            print(f"OK {ar:8s}: [Grand Ens v4] 予測 平均={bl.mean():.2f} (drift={drift_adj:+.2f}) 最小={bl.min():.2f} 最大={bl.max():.2f}")

        except Exception as e:
            warn.append(f"{ar}: {e}")
            print(f"WARN {ar}: {e}")

    if not rows:
        print("ERROR 全エリア失敗")
        return 1

    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "forecast_area_day1_recon_latest.csv", index=False, encoding="utf-8-sig")

    h_prev = load_area_pred_history()
    out_h = out.copy()
    out_h["target_datetime"] = pd.to_datetime(out_h["target_datetime"])
    h_merged = pd.concat([h_prev, out_h], ignore_index=True, sort=False) if not h_prev.empty else out_h
    h_merged = h_merged[h_merged["target_datetime"] >= day - pd.Timedelta(days=90)]
    h_merged = h_merged.drop_duplicates(["target_datetime", "area"], keep="last").sort_values(["target_datetime", "area"])
    h_merged.to_csv(PRED_HIST_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDAY1={day.date()} {out['area'].nunique()}エリア 成功 / 失敗{len(warn)}  システム予測のモデル={model}")
    print("出力:", OUT / "forecast_area_day1_recon_latest.csv")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["predict", "replay"], default="predict")
    ap.add_argument("--areas", nargs="+", default=AREAS)
    ap.add_argument("--oof", type=Path, default=Path(r"P:\エリアプライス2\学習\evaluation_area_recon_dedicated\oof.csv"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--system", type=Path, default=OUT / "forecast_system_day1_occto_latest.csv")
    ap.add_argument("--future", type=Path, default=INTEG)
    ap.add_argument("--gap-days", type=int, default=2)
    ap.add_argument("--blend-weight", type=float, default=0.5)
    a = ap.parse_args()

    sys.exit(predict(a))


if __name__ == "__main__":
    main()
