#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""エリア別DAY1: 新ブレンド予測 / 現行の予測 / 前日と同じ価格 を、JEPXの実績と同じ条件で比較する(読み取りのみ)。

 条件: 予測は「対象日の前日11:00(JEPX約定)までに作られたものだけ」。同じ対象コマに複数あれば最新を採用。

 3つがそろい、実績のあるコマだけで比べる。現行は forecast_day1_to_day7_history.csv のDAY1(エリア別)。

"""

import sys

from pathlib import Path

import numpy as np, pandas as pd

OUT = Path(r"P:\エリアプライス2\予測")

NEW = OUT / "forecast_area_day1_recon_history.csv"

OLD = Path(r"P:\エリアプライス2\web\forecast_day1_to_day7_history.csv")

JEPX = Path(r"P:\エリアプライス2\database\JPEXデータ\data\jepx_spot_summary_master.csv")

RES = OUT / "area_day1_compare_daily.csv"

AR = {"hokkaido": "北海道", "tohoku": "東北", "tokyo": "東京", "chubu": "中部", "hokuriku": "北陸", "kansai": "関西", "chugoku": "中国", "shikoku": "四国", "kyushu": "九州"}

def rd(p):

    for e in ("utf-8-sig", "cp932"):

        try: return pd.read_csv(p, encoding=e, low_memory=False)

        except UnicodeDecodeError: pass

def jst(s): return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

def pick(d, created, pcol, keys):

    d = d.assign(t=pd.to_datetime(d["target_datetime"], errors="coerce"), c=jst(d[created]), p=pd.to_numeric(d[pcol], errors="coerce")).dropna(subset=["t", "c", "p"])

    d = d[d["c"] <= d["t"].dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=11)]

    return d.sort_values("c").drop_duplicates(keys, keep="last")

def main():
    if not NEW.is_file():
        print(f"INFO {NEW.name} がまだ無いため実績比較をスキップします")
        return 0
    if not OLD.is_file():
        print(f"INFO {OLD.name} が無いため比較対象履歴なしとしてスキップします")
        return 0

    j = rd(JEPX); j["dt"] = pd.to_datetime(j["受渡日"]) + pd.to_timedelta((pd.to_numeric(j["時刻コード"]) - 1) * 30, unit="m"); j = j.drop_duplicates("dt").set_index("dt")

    act = pd.concat({a: pd.to_numeric(j[f"エリアプライス{jp}(円/kWh)"], errors="coerce") for a, jp in AR.items()}, axis=1)

    new = rd(NEW); old = rd(OLD)

    n = pick(new, "forecast_created_at", "predicted_price_yen_kwh", ["t", "area"])[["t", "area", "p"]].rename(columns={"p": "new"})

    cc = [c for c in old.columns if "created" in c.lower()][0]; ar = old[old["area"].astype(str).str.lower().isin(AR)]

    if "day" in ar.columns: ar = ar[pd.to_numeric(ar["day"], errors="coerce") == 1]

    if "predicted_price_direct_yen_kwh" in ar.columns:  # 置き換え後の履歴: 元の予測は専用の列に残っている

        ar = ar.assign(predicted_price_yen_kwh=pd.to_numeric(ar["predicted_price_direct_yen_kwh"], errors="coerce").fillna(pd.to_numeric(ar["predicted_price_yen_kwh"], errors="coerce")))

    o = pick(ar, cc, "predicted_price_yen_kwh", ["t", "area"])[["t", "area", "p"]].rename(columns={"p": "old"})

    print(f"新ブレンド(約定前に作成): {len(n):,}行  現行(DAY1・エリア別): {len(o):,}行  実績: 〜{act.dropna(how='all').index.max()}")

    m = n.merge(o, on=["t", "area"], how="left")

    m["actual"] = [act[a].get(t, np.nan) for t, a in zip(m["t"], m["area"])]; m["lag1"] = [act[a].get(t - pd.Timedelta(days=1), np.nan) for t, a in zip(m["t"], m["area"])]

    both = m.dropna(subset=["new", "actual", "lag1"])

    if both.empty:

        f = n["t"].min(); print(f"3つの予測と実績がそろったコマがまだありません。最初の対象日 {f.date() if pd.notna(f) else '-'} の実績は、{(f.normalize() - pd.Timedelta(days=1)).date() if pd.notna(f) else '-'} の11:00に確定します。"); return 0

    both = both.assign(date=both["t"].dt.date, e_new=both["new"] - both["actual"], e_old=both["old"] - both["actual"], e_lag=both["lag1"] - both["actual"])

    g = both.groupby(["date", "area"]).agg(コマ=("t", "size"), MAE_新=("e_new", lambda x: x.abs().mean()), MAE_現行=("e_old", lambda x: x.abs().mean()), MAE_前日=("e_lag", lambda x: x.abs().mean()), bias_新=("e_new", "mean"), bias_現行=("e_old", "mean"))

    g.round(3).to_csv(RES, encoding="utf-8-sig"); pd.set_option("display.width", 220)

    s = both.groupby("area").agg(日数=("date", "nunique"), MAE_新=("e_new", lambda x: x.abs().mean()), MAE_現行=("e_old", lambda x: x.abs().mean()), MAE_前日=("e_lag", lambda x: x.abs().mean()), bias_新=("e_new", "mean"), bias_現行=("e_old", "mean"))

    s["改善"] = s["MAE_現行"] - s["MAE_新"]; print(f"== エリア別({both['date'].nunique()}日 {len(both):,}コマ)\n" + s.round(3).to_string())

    a = both[["e_new", "e_old", "e_lag"]].abs().mean(); print(f"== 全エリア平均 MAE: 新={a['e_new']:.3f} 現行={a['e_old']:.3f} 前日={a['e_lag']:.3f}  新が現行より良いエリア: {int((s['MAE_新'] < s['MAE_現行']).sum())}/{len(s)}")

    d = both.groupby("date").agg(新=("e_new", lambda x: x.abs().mean()), 現行=("e_old", lambda x: x.abs().mean()), 前日=("e_lag", lambda x: x.abs().mean())); print("== 日別(全エリア平均)\n" + d.round(3).tail(14).to_string()); print("出力:", RES); return 0

if __name__ == "__main__": sys.exit(main())

