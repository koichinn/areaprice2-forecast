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
import argparse, re, sys, time
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
NOACT = r"^(?!システムプライス|area_price_yen_kwh|売り入札量|買い入札量|約定総量|pr_|premium_|dt_).*_lag\d+d$"

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


def apply_history_drift_correction(area: str, pred_prices: np.ndarray, target_dt: pd.Series, k=0.45, span=3) -> np.ndarray:
    """過去の予測履歴と実績から、直近残差の EWMA を算出して補正"""
    if not (PRED_HIST_CSV.exists() and JEPX_MASTER.exists()):
        return pred_prices

    try:
        hist = pd.read_csv(PRED_HIST_CSV, encoding="utf-8-sig")
        hist = hist[hist["area"] == area].copy()
        hist["target_datetime"] = pd.to_datetime(hist["target_datetime"])

        jepx = pd.read_csv(JEPX_MASTER, encoding="cp932")
        date_col = [c for c in jepx.columns if "受渡日" in str(c) or "date" in str(c).lower()][0]
        price_col = [c for c in jepx.columns if area in str(c).lower() or ("エリア" in str(c) and "価格" in str(c)) or "システムプライス" in str(c)][0]

        # 日次平均残差を計算
        # 予測履歴の最新数日分
        hist["_date"] = hist["target_datetime"].dt.normalize()
        recent_dates = sorted(hist["_date"].unique())[-10:]

        residuals = []
        for d in recent_dates:
            d_str = d.strftime("%Y/%m/%d")
            d_jepx = jepx[jepx[date_col] == d_str]
            if not d_jepx.empty and price_col in d_jepx.columns:
                act = pd.to_numeric(d_jepx[price_col], errors="coerce").mean()
                pred_m = hist[hist["_date"] == d]["predicted_price_yen_kwh"].mean()
                if np.isfinite(act) and np.isfinite(pred_m):
                    residuals.append(pred_m - act)

        if residuals:
            recent_res = residuals[-span:]
            weights = np.exp(np.linspace(-1, 0, len(recent_res)))
            weights /= weights.sum()
            avg_res = float(sum(r * w for r, w in zip(recent_res, weights)))
            # 補正適用 (上限 ±3円)
            adj = np.clip(k * avg_res, -3.0, 3.0)
            return pred_prices - adj
    except Exception:
        pass
    return pred_prices


def prep(area, v8dir=None):
    v8dir = v8dir or V8DIR
    A, acol, _, _ = E.load(v8dir / f"jepx_training_v8_price_regime_{area}_day1.parquet", "area", area)
    P, pcol, _, _ = E.load(v8dir / f"jepx_training_v8_{area}_premium_day1.parquet", "premium", f"{area}_premium")
    ca = [c for c in E.features(A) if not re.search(NOACT, c)]
    cp = [c for c in E.features(P) if not re.search(NOACT, c)]
    return A, acol, ca, P, pcol, cp


def fit_area(A, acol, ca, P, cp, train_end, gap_days=2):
    t = A.index.to_series().reset_index(drop=True)
    lim = train_end - pd.Timedelta(days=gap_days)
    ya = A["target"].to_numpy(float)
    alag = pd.to_numeric(A[acol], errors="coerce").to_numpy(float)
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

            if models_8 is not None and ar in models_8:
                m = models_8[ar]
                acol = m["acol"]
                ca = m["ca"]
                cp = m["cp"]
                lag = pd.to_numeric(xa[acol], errors="coerce").to_numpy(float)
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

                bl, rec = blend(d, pr, sp.to_numpy(float), a.blend_weight)

                # 3. 水準ドリフト補正 (EWMA)
                bl = apply_history_drift_correction(ar, bl, idx, k=0.45, span=3)

                n_a = m["n_direct"]
                n_p = m["n_premium"]
            else:
                A, acol, ca, P, pcol, cp = prep(ar)
                F = fit_area(A, acol, ca, P, cp, day, a.gap_days)
                miss = [c for c in ca if c not in xa.columns] + [c for c in cp if c not in xp.columns]
                if miss:
                    raise RuntimeError(f"将来の特徴量が足りません({len(miss)}列): {miss[:4]}")
                lag = pd.to_numeric(xa[acol], errors="coerce").to_numpy(float)
                if np.isnan(lag).any():
                    raise RuntimeError(f"{acol} が欠損です(前日の{ar}の価格がJEPXマスターにありません)")
                X_a = E.matrix(xa.reset_index(), ca)
                X_p = E.matrix(xp.reset_index(), cp)
                d, pr = predict_rows(F, X_a, lag, X_p)
                bl, rec = blend(d, pr, sp.to_numpy(float), a.blend_weight)
                bl = apply_history_drift_correction(ar, bl, idx, k=0.45, span=3)
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
                "forecast_created_at": now,
                "train_rows_direct": n_a,
                "train_rows_premium": n_p
            }))

            print(f"OK {ar:8s}: [Grand Ens v4] 予測 平均={bl.mean():.2f} 最小={bl.min():.2f} 最大={bl.max():.2f}")

        except Exception as e:
            warn.append(f"{ar}: {e}")
            print(f"WARN {ar}: {e}")

    if not rows:
        print("ERROR 全エリア失敗")
        return 1

    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "forecast_area_day1_recon_latest.csv", index=False, encoding="utf-8-sig")

    h = OUT / "forecast_area_day1_recon_history.csv"
    out.to_csv(h, mode="a", header=not h.exists(), index=False, encoding="utf-8-sig" if not h.exists() else "utf-8")

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
