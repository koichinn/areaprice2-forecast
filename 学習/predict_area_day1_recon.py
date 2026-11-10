#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""エリア別DAY1予測: 直接予測 + (専用System予測 + Premium予測) のブレンド。評価(evaluate_area_recon_dedicated.py)と同じ設定。

 predict: 最新データで学習し、DAY1の48コマを9エリア分予測 -> forecast_area_day1_recon_latest.csv(+history)

 replay : 評価のfoldを再現し、評価が保存した oof.csv と一致するか検証(同じ関数を使う)

"""

import argparse, re, sys

from datetime import datetime

from pathlib import Path

from types import SimpleNamespace

import numpy as np, pandas as pd

sys.path.insert(0, r"P:\エリアプライス2\学習")

import evaluate_area_premium_day1_clean_rolling as E
import predict_tokyo_day1_v4 as PTV4

INTEG = Path(r"P:\エリアプライス2\database\統合データ"); V8DIR = INTEG / "training_v8_price_regime"; OUT = Path(r"P:\エリアプライス2\予測")

AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]

NOACT = r"^(?!システムプライス|area_price_yen_kwh|売り入札量|買い入札量|約定総量|pr_|premium_|dt_).*_lag\d+d$"

HP = SimpleNamespace(max_iter=400, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=24, l2=2.0, random_state=42)

def prep(area, v8dir=None):

    v8dir = v8dir or V8DIR

    A, acol, _, _ = E.load(v8dir / f"jepx_training_v8_price_regime_{area}_day1.parquet", "area", area)

    P, pcol, _, _ = E.load(v8dir / f"jepx_training_v8_{area}_premium_day1.parquet", "premium", f"{area}_premium")

    ca = [c for c in E.features(A) if not re.search(NOACT, c)]; cp = [c for c in E.features(P) if not re.search(NOACT, c)]

    return A, acol, ca, P, pcol, cp

def fit_area(A, acol, ca, P, cp, train_end, gap_days=2):

    """学習: train_end - gap_days より前の行だけ。評価のrun_areaと同じ(直接=lag1残差, premium=そのもの)。"""

    t = A.index.to_series().reset_index(drop=True); lim = train_end - pd.Timedelta(days=gap_days)

    ya = A["target"].to_numpy(float); alag = pd.to_numeric(A[acol], errors="coerce").to_numpy(float); yp = P.reindex(A.index)["target"].to_numpy(float)

    trb = (t < lim).to_numpy(); tr_a = trb & ~np.isnan(alag) & ~np.isnan(ya); tr_p = trb & ~np.isnan(yp)

    aX = E.matrix(A, ca); pX = E.matrix(P.reindex(A.index), cp)

    return SimpleNamespace(aX=aX, pX=pX, ya=ya, alag=alag, yp=yp, tr_a=tr_a, tr_p=tr_p, n_a=int(tr_a.sum()), n_p=int(tr_p.sum()))

def predict_rows(F, X_a, lag_a, X_p):

    """学習行(F)で学習し、X_a/X_p(新しい行)を予測。E.fit_predict(X,y,tr,te,a) は X の te 行を予測するので、学習+新しい行を縦に結合して渡す。"""

    n = len(F.ya); Xa = np.vstack([F.aX, X_a]); Xp = np.vstack([F.pX, X_p]); te = np.zeros(n + len(X_a), bool); te[n:] = True

    ya_full = np.concatenate([F.ya - F.alag, np.zeros(len(X_a))]); yp_full = np.concatenate([F.yp, np.zeros(len(X_p))])

    tr_a = np.concatenate([F.tr_a, np.zeros(len(X_a), bool)]); tr_p = np.concatenate([F.tr_p, np.zeros(len(X_p), bool)])

    direct = lag_a + E.fit_predict(Xa, ya_full, tr_a, te, HP); prem = E.fit_predict(Xp, yp_full, tr_p, te, HP); return direct, prem

def blend(direct, prem, sysp, w=0.5):

    rec = sysp + prem; return w * direct + (1 - w) * rec, rec

def replay(a):

    oof = pd.read_csv(a.oof, encoding="utf-8-sig"); oof["target_datetime"] = pd.to_datetime(oof["target_datetime"]); worst = 0.0; ok_all = True

    for ar in a.areas:

        o = oof[(oof["area"] == ar)]; fold = sorted(o["fold"].unique())[-1]; g = o[o["fold"] == fold].set_index("target_datetime"); t0 = pd.Timestamp(fold.split("~")[0])

        A, acol, ca, P, pcol, cp = prep(ar); F = fit_area(A, acol, ca, P, cp, t0, a.gap_days); idx = g.index.intersection(A.index)

        tp = A.index.get_indexer(idx); Pidx = P.reindex(A.index); X_a = E.matrix(A, ca)[tp]; X_p = E.matrix(Pidx, cp)[tp]; lag = pd.to_numeric(A[acol], errors="coerce").to_numpy(float)[tp]

        d, pr = predict_rows(F, X_a, lag, X_p); sysp = g.loc[idx, "system_pred"].to_numpy(float); bl, rec = blend(d, pr, sysp, a.blend_weight)

        dif = {k: float(np.nanmax(np.abs(v - g.loc[idx, c].to_numpy(float)))) for k, v, c in (("direct", d, "direct"), ("premium", pr, "premium_pred"), ("blend", bl, "blend"))}

        ok = all(v < 1e-6 for v in dif.values()); ok_all &= ok; worst = max(worst, max(dif.values()))

        print(f"{ar}: fold {fold} 行{len(idx)} 学習行 直接={F.n_a:,} premium={F.n_p:,} 最大差 直接={dif['direct']:.2g} premium={dif['premium']:.2g} blend={dif['blend']:.2g} -> {'一致' if ok else '不一致'}")

    print("RESULT:", "全て一致" if ok_all else f"不一致あり(最大差{worst:.3g})"); return 0 if ok_all else 2

def predict(a):

    sysf = pd.read_csv(a.system, encoding="utf-8-sig"); sysf["target_datetime"] = pd.to_datetime(sysf["target_datetime"]); day = pd.Timestamp(a.date) if a.date else sysf["target_datetime"].min().normalize()

    idx = pd.date_range(day, periods=48, freq="30min"); sp = sysf.drop_duplicates("target_datetime", keep="last").set_index("target_datetime")["predicted_system_price_yen_kwh"].reindex(idx)

    if sp.isna().any(): print(f"ERROR 専用システム予測が{day.date()}の48コマ揃っていません(欠損{int(sp.isna().sum())})"); return 1

    model = str(sysf["model_used"].iloc[-1]); fa = pd.read_parquet(a.future / "future_area_day1_area.parquet"); fp = pd.read_parquet(a.future / "future_area_day1_premium.parquet")

    for x in (fa, fp): x["target_datetime"] = pd.to_datetime(x["target_datetime"])

    rows, warn = [], []; now = datetime.now().astimezone().isoformat()

    for ar in AREAS:

        try:
            if ar == "tokyo":
                print(">> tokyo は新アンサンブルモデル v4_tokyo_super を適用します")
                try:
                    tok_df = PTV4.predict_tokyo_day1(day, sp.to_numpy(float), iter_scale=0.35)
                    rows.append(tok_df)
                    print(f"OK tokyo (v4_tokyo_super): 予測 平均={tok_df['predicted_price_yen_kwh'].mean():.2f} 最小={tok_df['predicted_price_yen_kwh'].min():.2f} 最大={tok_df['predicted_price_yen_kwh'].max():.2f}")
                    continue
                except Exception as ex_tok:
                    print(f"WARN tokyo v4_tokyo_super 失敗のため従来手法へフォールバック: {ex_tok}")
            A, acol, ca, P, pcol, cp = prep(ar); F = fit_area(A, acol, ca, P, cp, day, a.gap_days)

            xa = fa[fa["area"] == ar].set_index("target_datetime").reindex(idx); xp = fp[fp["area"] == ar].set_index("target_datetime").reindex(idx)

            miss = [c for c in ca if c not in xa.columns] + [c for c in cp if c not in xp.columns]

            if miss: raise RuntimeError(f"将来の特徴量が足りません({len(miss)}列): {miss[:4]}")

            lag = pd.to_numeric(xa[acol], errors="coerce").to_numpy(float)

            if np.isnan(lag).any(): raise RuntimeError(f"{acol} が欠損です(前日の{ar}の価格がJEPXマスターにありません)")

            X_a = E.matrix(xa.reset_index(), ca); X_p = E.matrix(xp.reset_index(), cp); d, pr = predict_rows(F, X_a, lag, X_p); bl, rec = blend(d, pr, sp.to_numpy(float), a.blend_weight)

            if np.nanmin(bl) < -5 or np.nanmax(bl) > 300: raise RuntimeError(f"価格が範囲外({np.nanmin(bl):.1f}〜{np.nanmax(bl):.1f})")

            rows.append(pd.DataFrame({"target_datetime": idx, "area": ar, "predicted_price_yen_kwh": bl, "direct": d, "reconstructed": rec, "system_pred": sp.to_numpy(float), "premium_pred": pr,

                                      "anchor_lag1d": lag, "model_used": f"blend0.5(direct+system:{model})", "forecast_created_at": now, "train_rows_direct": F.n_a, "train_rows_premium": F.n_p}))

            print(f"OK {ar}: 学習行 直接={F.n_a:,} premium={F.n_p:,} 予測 平均={bl.mean():.2f} 最小={bl.min():.2f} 最大={bl.max():.2f}")

        except Exception as e: warn.append(f"{ar}: {e}"); print(f"WARN {ar}: {e}")

    if not rows: print("ERROR 全エリア失敗"); return 1

    out = pd.concat(rows, ignore_index=True); OUT.mkdir(parents=True, exist_ok=True); out.to_csv(OUT / "forecast_area_day1_recon_latest.csv", index=False, encoding="utf-8-sig")

    h = OUT / "forecast_area_day1_recon_history.csv"; out.to_csv(h, mode="a", header=not h.exists(), index=False, encoding="utf-8-sig" if not h.exists() else "utf-8")

    print(f"DAY1={day.date()} {out['area'].nunique()}エリア 成功 / 失敗{len(warn)}  システム予測のモデル={model}"); print("出力:", OUT / "forecast_area_day1_recon_latest.csv"); return 0

def main():

    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["predict", "replay"], default="predict"); ap.add_argument("--areas", nargs="+", default=AREAS)

    ap.add_argument("--oof", type=Path, default=Path(r"P:\エリアプライス2\学習\evaluation_area_recon_dedicated\oof.csv")); ap.add_argument("--date", default=None)

    ap.add_argument("--system", type=Path, default=OUT / "forecast_system_day1_occto_latest.csv"); ap.add_argument("--future", type=Path, default=INTEG)

    ap.add_argument("--gap-days", type=int, default=2); ap.add_argument("--blend-weight", type=float, default=0.5); a = ap.parse_args()

    sys.exit(replay(a) if a.mode == "replay" else predict(a))

if __name__ == "__main__": main()

