#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""OCCTO翌々日系列の学習用8列(occto_da8.parquet)を、月次CSVから作り直す(毎日の更新から呼ぶ)。

 1) 月次CSVを、前回の最終日の月初から今日まで取り直す  2) 9エリア平均の8列を作る

 3) 検証(行数・最終日が後退しない)に通れば置き換える。前の版は .prev.parquet に残す。

"""

import argparse, shutil, subprocess, sys

from datetime import date

from pathlib import Path

import numpy as np, pandas as pd

try:

    sys.stdout.reconfigure(errors="replace")

except Exception:

    pass

B = Path(r"P:\エリアプライス2\database\occtoデータ")

OUT = Path(r"P:\エリアプライス2\学習\evaluation_system_day1_levers_da\occto_da8.parquet")

AR = ["北海道", "東北", "東京", "中部", "北陸", "関西", "中国", "四国", "九州"]

M = {"エリア需要(MW)": "area_demand_mw", "エリア供給力(MW)": "area_supply_mw", "エリア予備力(MW)": "area_reserve_mw",

     "広域ブロック需要(MW)": "block_demand_mw", "広域ブロック供給力(MW)": "block_supply_mw", "広域ブロック予備力(MW)": "block_reserve_mw",

     "広域予備率(%)": "wide_reserve_rate_pct", "広域使用率(%)": "wide_utilization_pct"}

def rd(p):

    for enc in ("utf-8-sig", "cp932"):

        try:

            sk = 1 if "UPDATE" in open(p, encoding=enc).readline() else 0

            d = pd.read_csv(p, encoding=enc, skiprows=sk); d.columns = [str(c).strip().rstrip(".") for c in d.columns]; return d

        except UnicodeDecodeError:

            pass

def build(base):

    fs = sorted((base / "bulk_downloads" / "wide_block_dayaftertomorrow").glob("*.csv"), key=lambda p: p.stat().st_mtime)

    if not fs: return pd.DataFrame()

    d = pd.concat([rd(p) for p in fs]); d = d[d["エリア名"].isin(AR)]

    t = d["時刻"].astype(str).str.replace(r"^(\d):", r"0\1:", regex=True)

    dt = pd.to_datetime(d["対象年月日"].astype(str).str.replace("-", "/")) + pd.to_timedelta(pd.to_numeric(t.str[:2]) * 60 + pd.to_numeric(t.str[3:5]) - 30, unit="m")

    x = d.assign(dt=dt).drop_duplicates(["dt", "エリア名"], keep="last")

    use = [c for c in M if c in x.columns]

    return x.groupby("dt")[use].mean().rename(columns=M)

def main():

    ap = argparse.ArgumentParser(); ap.add_argument("--no-fetch", action="store_true")

    ap.add_argument("--base", type=Path, default=B); ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args()

    old = pd.read_parquet(a.out) if a.out.exists() else None

    om = pd.to_datetime(old["target_datetime"]).max() if old is not None else None

    start = (om - pd.Timedelta(days=7)).replace(day=1) if om is not None else pd.Timestamp("2025-04-01")

    if not a.no_fetch:

        cmd = [sys.executable, str(a.base / "backfill_occto_forecast_bulk_v2.py"), "--from-date", start.strftime("%Y-%m-%d"), "--to-date", date.today().isoformat()]

        r = subprocess.run(cmd, cwd=str(a.base), capture_output=True, text=True, encoding="cp932", errors="replace")

        print("月次CSV取得: exit=%d  %s" % (r.returncode, " | ".join((r.stdout or "").strip().splitlines()[-2:])))

        if r.returncode != 0: print("WARN 取得に失敗。既にある月次CSVで作り直します:", (r.stderr or "")[-160:].replace("\n", " "))

    new = build(a.base)

    if new.empty: print("ERROR 作成結果が空です"); return 1

    nm = new.index.max()

    if old is not None:
        o_prev = old.assign(target_datetime=pd.to_datetime(old["target_datetime"])).set_index("target_datetime")
        o_prev.index.name = "dt"
        o_prev_clean = o_prev.rename(columns=lambda c: c[len("pre1100_occto_"):] if c.startswith("pre1100_occto_") else c)
        if len(new) < len(old) * 0.99:
            print(f"   INFO 差分月次データ({len(new):,}行)を既存履歴({len(old):,}行)とマージします")
            common_cols = [c for c in new.columns if c in o_prev_clean.columns]
            older_rows = o_prev_clean.loc[~o_prev_clean.index.isin(new.index), common_cols]
            new = pd.concat([older_rows, new[common_cols]]).sort_index()
            nm = new.index.max()

        if nm < om: print(f"ERROR 最終日が後退します({nm} < {om})。置き換えません"); return 1

        if len(new) < len(old) * 0.99: print(f"ERROR 行数が減ります({len(new):,} < {len(old):,})。置き換えません"); return 1

        o = old.assign(target_datetime=pd.to_datetime(old["target_datetime"])).set_index("target_datetime")

        j = new.index.intersection(o.index); j = j[j <= om - pd.Timedelta(days=3)]

        cols = [c for c in new.columns if "pre1100_occto_" + c in o.columns]

        if len(j) and cols:

            n = sum(int((~np.isclose(new.loc[j, c], o.loc[j, "pre1100_occto_" + c], rtol=1e-6, atol=1e-6)).sum()) for c in cols)

            print(f"既存期間の値の変化: {n}/{len(j) * len(cols)}セル ({n / (len(j) * len(cols)) * 100:.2f}%)")

    out = new.rename(columns=lambda c: "pre1100_occto_" + c).reset_index().rename(columns={"dt": "target_datetime"})

    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.out.exists(): shutil.copy2(a.out, a.out.with_name(a.out.stem + ".prev.parquet"))

    tmp = a.out.with_name(a.out.stem + ".tmp.parquet"); out.to_parquet(tmp, index=False); tmp.replace(a.out)

    print(f"OK occto_da8: {len(out):,}行 {new.index.min()}〜{nm}  (前回 {0 if old is None else len(old):,}行 〜{om})")

    return 0

if __name__ == "__main__": sys.exit(main())

