#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""エリア別DAY1を、新ブレンド予測(predict_area_day1_recon.py の出力)に置き換える。
 対象: forecast_day1_to_day7_latest.csv の 9エリア x day=1 の行だけ。システム価格とDAY2-7は変更しない。
 元の予測は predicted_price_direct_yen_kwh 列に残す。エリアごとに条件を判定し、外れたエリアだけ置き換えない。
 条件: 新ブレンドが今回の実行で作られた / DAY1の48コマが揃う / 欠損なし / 価格が -5〜300円
 --restore: 置き換え前の状態に戻す
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass
OUT = Path(r"P:\エリアプライス2\予測")
AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]
def rd(p):
    for e in ("utf-8-sig", "cp932"):
        try: return pd.read_csv(p, encoding=e, low_memory=False)
        except UnicodeDecodeError: pass
def prepare(a):
    bak = a.latest.with_name(a.latest.stem + "_before_area.csv")
    if a.restore:
        if not bak.exists(): print("ERROR 置き換え前のバックアップがありません"); return 1
        shutil.copy2(bak, a.latest); print("復元しました:", a.latest); return 0
    if not a.latest.exists() or not a.recon.exists(): print("SKIP 予測ファイルがありません:", a.latest if not a.latest.exists() else a.recon); return 2
    x = rd(a.latest); d = rd(a.recon)
    if not {"area", "day", "target_datetime", "predicted_price_yen_kwh"} <= set(x.columns): print("SKIP 想定外の列構成:", list(x.columns)[:12]); return 2
    d["target_datetime"] = pd.to_datetime(d["target_datetime"]); age = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(d["forecast_created_at"], utc=True).max()).total_seconds() / 3600
    if age > a.max_age_hours: print(f"SKIP 置き換えません: 新ブレンドが古い({age:.1f}時間前)"); return 2
    xa = x["area"].astype(str).str.lower(); xd = pd.to_numeric(x["day"], errors="coerce"); xt = pd.to_datetime(x["target_datetime"])
    x["predicted_price_direct_yen_kwh"] = pd.to_numeric(x["predicted_price_direct_yen_kwh"], errors="coerce") if "predicted_price_direct_yen_kwh" in x.columns else np.nan
    x["system_day1_model"] = x["system_day1_model"].astype(object) if "system_day1_model" in x.columns else pd.Series([None] * len(x), index=x.index, dtype=object)
    return x, d, xa, xd, xt, bak
def apply(a, x, d, xa, xd, xt, bak):
    plans, skipped = {}, []
    for ar in AREAS:
        m = ((xa == ar) & (xd == 1)).to_numpy(); da = d[d["area"].astype(str).str.lower() == ar].drop_duplicates("target_datetime", keep="last").set_index("target_datetime"); why = []
        p = pd.to_numeric(da["predicted_price_yen_kwh"], errors="coerce") if len(da) else pd.Series(dtype=float)
        if m.sum() != 48: why.append(f"取り込み元が{int(m.sum())}行(48行のはず)")
        if len(da) != 48 or set(da.index) != set(xt[m]): why.append("対象コマが一致しない")
        if not why and p.isna().any(): why.append("新ブレンドに欠損")
        if not why and (p.min() < -5 or p.max() > 300): why.append(f"価格が範囲外({p.min():.1f}〜{p.max():.1f})")
        if why: skipped.append(f"{ar}: " + "・".join(why)); continue
        plans[ar] = (m, p, str(da["model_used"].iloc[-1]))
    for s in skipped: print("SKIP", s)
    if not plans: print("SKIP どのエリアも置き換えませんでした"); return 2
    if any(x.loc[m, "predicted_price_direct_yen_kwh"].isna().all() for m, _, _ in plans.values()): shutil.copy2(a.latest, bak)
    diff = {}
    for ar, (m, p, mdl) in plans.items():
        cur = pd.to_numeric(x.loc[m, "predicted_price_yen_kwh"], errors="coerce"); x.loc[m, "predicted_price_direct_yen_kwh"] = x.loc[m, "predicted_price_direct_yen_kwh"].fillna(cur).to_numpy()
        x.loc[m, "predicted_price_yen_kwh"] = p.reindex(xt[m]).to_numpy(); x.loc[m, "system_day1_model"] = mdl; diff[ar] = float((x.loc[m, "predicted_price_yen_kwh"] - x.loc[m, "predicted_price_direct_yen_kwh"]).mean())
    x.to_csv(a.latest, index=False, encoding="utf-8-sig"); pq = a.latest.with_suffix(".parquet")
    if pq.exists():
        try: x.to_parquet(pq, index=False)
        except Exception as e: print("WARN parquetは更新できませんでした:", str(e)[:80])
    print(f"OK {len(plans)}/9エリアのDAY1(各48コマ)を新ブレンドに置き換えました。元の予測との差の平均: " + " ".join(f"{k}={v:+.1f}" for k, v in diff.items()))
    print(f"   システム価格とDAY2-7は変更なし。元の予測: predicted_price_direct_yen_kwh 列 / バックアップ: {bak.name}"); return 0
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-age-hours", type=float, default=6.0); ap.add_argument("--restore", action="store_true")
    ap.add_argument("--latest", type=Path, default=OUT / "forecast_day1_to_day7_latest.csv"); ap.add_argument("--recon", type=Path, default=OUT / "forecast_area_day1_recon_latest.csv"); a = ap.parse_args()
    r = prepare(a)
    return r if isinstance(r, int) else apply(a, *r)
if __name__ == "__main__": sys.exit(main())
