#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""エリア別DAY1(直接/プレミアム)の特徴量を、予測日の48コマ分、学習データと同じ関数で作る。

 既存の関数をそのまま読み込む(v6:build_lags/add_metadata, v8:engineer, プレミアム:build_one, 評価:add_dt, 気象:engineer)。

 replay: 過去の日を「実績を伏せた予測日」として作り直し、学習データの同じ行(モデルが使う列)と一致するか検証

 fetch : DAY1(11:00ルール)の48コマを、Open-Meteoの予報で作る -> future_area_day1_{area,premium}.parquet

"""

import argparse, importlib.util, re, sys, tempfile

from datetime import date, datetime, timedelta

from pathlib import Path

from types import SimpleNamespace

import numpy as np, pandas as pd

L = Path(r"P:\エリアプライス2\学習"); INTEG = Path(r"P:\エリアプライス2\database\統合データ")

sys.path.insert(0, str(L))

AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]

INPUT = INTEG / "past_integrated_9area_30min_occto_weather_day1.parquet"; V8DIR = INTEG / "training_v8_price_regime"

LAGS = [1, 2, 7, 14, 28]

NOACT = r"^(?!システムプライス|area_price_yen_kwh|売り入札量|買い入札量|約定総量|pr_|premium_|dt_).*_lag\d+d$"

def find(name):

    for p in (INTEG / name, L / name):

        if p.is_file(): return p

    raise FileNotFoundError(name)

def mod(path, name):

    sp = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

FALLBACK_INPUT = INTEG / "past_integrated_9area_30min.parquet"
def prep(v6, path):
    if not path.is_file() and FALLBACK_INPUT.is_file():
        print(f"   INFO {path.name} が無いため {FALLBACK_INPUT.name} を使用します")
        path = FALLBACK_INPUT
    elif path.is_file() and FALLBACK_INPUT.is_file() and FALLBACK_INPUT.stat().st_mtime > path.stat().st_mtime + 60:
        print(f"   INFO より新しい {FALLBACK_INPUT.name} を優先して使用します")
        path = FALLBACK_INPUT

    d = v6.read(path); d["datetime"] = pd.to_datetime(d["datetime"], errors="raise").dt.floor("30min"); d["area"] = v6.normalize_area(d["area"])

    return d[d["area"].isin(AREAS)].sort_values(v6.KEYS).drop_duplicates(v6.KEYS, keep="last")

def wf_frame(wfm, hist, areas):

    return pd.concat([wfm.engineer(hist, ar, 1).assign(area=ar).rename(columns={"target_datetime": "datetime"}) for ar in areas], ignore_index=True)

JEPX_MASTER = Path(r"P:\エリアプライス2\database\JPEXデータ\data\jepx_spot_summary_master.csv")

AREA_JP = {"北海道": "hokkaido", "東北": "tohoku", "東京": "tokyo", "中部": "chubu", "北陸": "hokuriku", "関西": "kansai", "中国": "chugoku", "四国": "shikoku", "九州": "kyushu"}

SYSCOL = "システムプライス(円/kWh)"

_JX = {}

def load_jepx():

    """JEPXマスターから、エリア価格とシステム価格を(datetime, area)で引けるようにする。"""

    if "jx" in _JX: return _JX["jx"]

    j = None

    for enc in ("utf-8-sig", "cp932"):

        try: j = pd.read_csv(JEPX_MASTER, encoding=enc, low_memory=False); break

        except (UnicodeDecodeError, FileNotFoundError): pass

    if j is None: _JX["jx"] = None; return None

    j["datetime"] = pd.to_datetime(j["受渡日"]) + pd.to_timedelta((pd.to_numeric(j["時刻コード"]) - 1) * 30, unit="m"); j = j.drop_duplicates("datetime", keep="last"); parts = []

    for jp, en in AREA_JP.items():

        c = f"エリアプライス{jp}(円/kWh)"

        if c in j.columns: parts.append(pd.DataFrame({"datetime": j["datetime"], "area": en, "area_price_yen_kwh": pd.to_numeric(j[c], errors="coerce"), SYSCOL: pd.to_numeric(j[SYSCOL], errors="coerce")}))

    _JX["jx"] = pd.concat(parts, ignore_index=True) if parts else None; return _JX["jx"]

def fill_prices(d2, day):

    """予測日より前の行で、価格が空のところをJEPXマスターの確定済み価格で埋める(予測日そのものは埋めない)。"""

    jx = load_jepx()

    if jx is None: print("WARN JEPXマスターを読めません。価格の空きは埋めません"); return d2

    m = d2[["datetime", "area"]].merge(jx, on=["datetime", "area"], how="left", validate="many_to_one"); ok = (d2["datetime"] < day).to_numpy(); n0 = 0

    for c in ("area_price_yen_kwh", SYSCOL):

        v = m[c].to_numpy(float)

        if c not in d2.columns: d2[c] = np.nan

        need = d2[c].isna().to_numpy() & ok & ~np.isnan(v); n0 += int(need.sum()); d2[c] = np.where(need, v, d2[c].to_numpy(float))

    if n0: print(f"   履歴の価格の空きをJEPXマスターで補完: {n0:,}セル")

    return d2

def build_day(M, d, actual, targets, wf_all, day, areas, win=45):

    v6, v8, prem, E = M.v6, M.v8, M.prem, M.E

    idx = pd.date_range(day, periods=48, freq="30min")

    fut = pd.DataFrame({"datetime": np.tile(idx, len(areas)), "area": np.repeat(areas, 48)})

    h0 = d[(d["datetime"] >= day - pd.Timedelta(days=win)) & (d["datetime"] < day)]

    gi = pd.date_range(h0["datetime"].max() + pd.Timedelta(minutes=30), day - pd.Timedelta(minutes=30), freq="30min") if len(h0) else pd.DatetimeIndex([])

    gap = [pd.DataFrame({"datetime": np.tile(gi, len(areas)), "area": np.repeat(areas, len(gi))})] if len(gi) else []

    d2 = fill_prices(pd.concat([h0] + gap + [fut], ignore_index=True), day)

    d2 = d2.drop(columns=[c for c in d2.columns if c.startswith("wf_")]).merge(wf_all, on=["datetime", "area"], how="left", validate="one_to_one")

    s = d2.drop_duplicates("datetime")[["datetime", targets[1]]].rename(columns={"datetime": "target_datetime", targets[1]: "target"})

    s["target"] = pd.to_numeric(s["target"], errors="coerce"); oa, op = [], []

    with tempfile.TemporaryDirectory() as td:

        td = Path(td)

        for ar in areas:

            da = d2[d2["area"] == ar]

            lg = v6.build_lags(da, actual, targets, LAGS).merge(da[v6.KEYS + targets], on=v6.KEYS, how="left", validate="one_to_one")

            out, _ = v8.engineer(v6.add_metadata(lg, v6.AREA_TARGET, f"{ar}_area_price_day1", "10:30:00"), f"{ar}_day1")

            out = out.sort_values("target_datetime").reset_index(drop=True); pth = td / f"a_{ar}.parquet"; out.to_parquet(pth, index=False)

            prem.build_one(ar, pth, s, SimpleNamespace(output_dir=td, csv=False))

            pr = pd.read_parquet(td / f"jepx_training_v8_{ar}_premium_day1.parquet")

            for lst, x in ((oa, out), (op, pr)):

                x = x.sort_values("target_datetime").reset_index(drop=True); x, _, _ = E.add_dt(x); lst.append(x[x["target_datetime"].isin(idx)].copy())

    return pd.concat(oa, ignore_index=True), pd.concat(op, ignore_index=True)

def setup():

    M = SimpleNamespace(); M.v6 = mod(find("build_jepx_training_dataset_occto_weather_v6.py"), "v6m"); M.v8 = mod(find("build_jepx_training_dataset_price_regime_v8.py"), "v8m")

    M.prem = mod(find("build_area_premium_dataset_day1.py"), "premm"); import evaluate_area_premium_day1_clean_rolling as E; M.E = E

    M.fw = mod(L / "build_future_weather_features_system.py", "fwm"); return M

def sets(M, d):

    system = M.v6.find_system_target(d, None); targets = [M.v6.AREA_TARGET, system]; actual, _ = M.v6.numeric_feature_sets(d, targets); return targets, actual

def compare(fut, tr, cols, label):

    f = fut.set_index("target_datetime"); j = f.index.intersection(tr.index); miss = [c for c in cols if c not in f.columns]; bad = {}

    for c in cols:

        if c in f.columns:

            x, y = pd.to_numeric(f.loc[j, c], errors="coerce").to_numpy(float), pd.to_numeric(tr.loc[j, c], errors="coerce").to_numpy(float)

            n = int((~np.isclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True)).sum())

            if n: bad[c] = n

    print(f"   {label}: 比較{len(j)}行 特徴量{len(cols)}本 足りない列={len(miss)} {miss[:3]} 値が違う列={len(bad)} {dict(list(bad.items())[:5])}")

    return len(miss) + len(bad)

def replay(a):

    M = setup(); wfm = M.fw.load(M.fw.WDIR / "build_weather_forecast_features_day1.py", "wfm"); E = M.E

    d = prep(M.v6, INPUT); targets, actual = sets(M, d)

    hist = pd.read_parquet(M.fw.WDIR / "forecast_history" / "weather_forecast_history_30min.parquet"); hist["target_datetime"] = pd.to_datetime(hist["target_datetime"]); hist = hist[hist["forecast_horizon_days"] == 1]

    tr = {}

    for ar in a.areas:

        A, _, _, _ = E.load(V8DIR / f"jepx_training_v8_price_regime_{ar}_day1.parquet", "area", ar); P, _, _, _ = E.load(V8DIR / f"jepx_training_v8_{ar}_premium_day1.parquet", "premium", f"{ar}_premium")

        tr[ar] = (A, [c for c in E.features(A) if not re.search(NOACT, c)], P, [c for c in E.features(P) if not re.search(NOACT, c)])

    last = tr[a.areas[0]][0].index.max().normalize(); bad = 0

    for k in range(a.replay_days):

        day = last - pd.Timedelta(days=a.back + k); h = hist[(hist["target_datetime"] >= day - pd.Timedelta(days=1)) & (hist["target_datetime"] < day + pd.Timedelta(days=1))]

        fa, fp = build_day(M, d, actual, targets, wf_frame(wfm, h, a.areas), day, a.areas); print(f"== 予測日 {day.date()} (学習データの最終日 {last.date()} の{a.back + k}日前)")

        for ar in a.areas:

            A, ca, P, cp = tr[ar]; idx = pd.date_range(day, periods=48, freq="30min")

            bad += compare(fa[fa["area"] == ar], A.loc[A.index.isin(idx)], ca, f"{ar} 直接用"); bad += compare(fp[fp["area"] == ar], P.loc[P.index.isin(idx)], cp, f"{ar} プレミアム用")

    print("RESULT:", "全て一致" if bad == 0 else f"違いあり({bad}件)"); return 0 if bad == 0 else 2

def fetch(a):

    M = setup(); FW = M.fw; FP = mod(L / "build_future_price_features_system.py", "fpm"); wfm = FW.load(FW.WDIR / "build_weather_forecast_features_day1.py", "wfm")

    bf = FW.load(FW.WDIR / "backfill_open_meteo_forecast_history.py", "bf"); day = pd.Timestamp(a.date) if a.date else FP.day1_date(datetime.now())

    sess = FW.requests.Session(); sess.headers["User-Agent"] = "jepx-weather-future/1.0"; t0 = date.today(); fr = []

    for ar in AREAS: print("取得中:", ar); fr.append(FW.fetch_area(sess, bf, ar, t0, t0 + timedelta(days=8), not a.no_ssl_verify))

    wf = wf_frame(wfm, bf.to30(pd.concat(fr, ignore_index=True)), AREAS); d = prep(M.v6, INPUT); targets, actual = sets(M, d)

    fa, fp = build_day(M, d, actual, targets, wf, day, AREAS); print(f"DAY1={day.date()} 直接用{fa.shape} プレミアム用{fp.shape}")

    for n, x in (("直接用", fa), ("プレミアム用", fp)):

        w = [c for c in x.columns if c.startswith("wf_")]; lg = [c for c in x.columns if re.match(r"(area_price_yen_kwh|システムプライス.*)_lag\d+d$", c)]

        print(f"  {n}: wf_列={len(w)} 欠損のあるwf_列={int(x[w].isna().any().sum())} 価格ラグ列={len(lg)} 欠損のある価格ラグ列={int(x[lg].isna().any().sum())}")

    a.out_dir.mkdir(parents=True, exist_ok=True); fa.to_parquet(a.out_dir / "future_area_day1_area.parquet", index=False); fp.to_parquet(a.out_dir / "future_area_day1_premium.parquet", index=False); print("出力:", a.out_dir); return 0

def main():

    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["replay", "fetch"], default="fetch"); ap.add_argument("--areas", nargs="+", default=AREAS)

    ap.add_argument("--replay-days", type=int, default=2); ap.add_argument("--back", type=int, default=7); ap.add_argument("--date", default=None)

    ap.add_argument("--out-dir", type=Path, default=INTEG); ap.add_argument("--no-ssl-verify", action="store_true"); a = ap.parse_args()

    sys.exit(replay(a) if a.mode == "replay" else fetch(a))

if __name__ == "__main__": main()

