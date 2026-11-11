#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""システムDAY1用の将来気象特徴量(wf_ 133列)を、学習データ(natwf)と同じ計算で作る。

 fetch : Open-Meteo予報API(最新ラン)から9エリアx12変数を取得し、to30→engineer→9エリア統計

 replay: 予報履歴(previous_day1)の直近日を同じ処理に通し、学習データのwf_列と一致するか検証

"""

import argparse, importlib.util, sys

from datetime import date, timedelta

from pathlib import Path

import numpy as np, pandas as pd

try:

    import truststore; truststore.inject_into_ssl()

except ImportError:

    pass

import requests

AREAS = ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]

NAT_VARS = ["wf_temperature_c", "wf_cdd_24", "wf_cdd_28", "wf_hdd_15", "wf_humidity_pct", "wf_cloud_cover_pct",

            "wf_wind_speed_10m", "wf_wind_power_proxy", "wf_shortwave_radiation_wm2", "wf_solar_effective_wm2",

            "wf_precipitation_mm_30min", "wf_direct_radiation_wm2"]

PER_VARS = ["wf_temperature_c", "wf_solar_effective_wm2", "wf_wind_speed_10m"]

WDIR = Path(r"P:\エリアプライス2\database\天気9エリア")

DB = Path(r"P:\エリアプライス2\database\統合データ")

API_URL = "https://api.open-meteo.com/v1/forecast"

def load(path, name):

    sp = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(sp)

    sp.loader.exec_module(m); return m

def fetch_area(sess, bf, area, start, end, verify, max_retries=4):
    import time
    st, lat, lon = bf.AREAS[area]
    pr = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
          "hourly": ",".join(bf.BASE_VARS), "timezone": "Asia/Tokyo"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = sess.get(API_URL, params=pr, timeout=(30, 180), verify=verify)
            if not r.ok:
                raise RuntimeError(f"{area}: HTTP {r.status_code}: {r.text[:300]}")
            h = r.json().get("hourly", {})
            if "time" not in h:
                raise RuntimeError(f"{area}: hourlyなし")
            raw = pd.DataFrame(h); x = pd.DataFrame({"target_datetime": pd.to_datetime(raw.pop("time"))})
            for v in bf.BASE_VARS: x[v] = pd.to_numeric(raw[v], errors="coerce") if v in raw else np.nan
            x.insert(1, "area", area); x.insert(2, "station", st); x.insert(3, "latitude", lat); x.insert(4, "longitude", lon)
            x["forecast_horizon_days"] = 1; x["run_lag_days"] = 1; x["issue_time_inferred"] = pd.Timestamp(date.today())
            x["issue_time_is_inferred"] = 1; x["source_api"] = "open-meteo_forecast_latest"; x["source_model"] = "best_match"
            return x
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait_sec = 8 * (2 ** (attempt - 1))
                print(f"  [WARN] {area} 取得失敗 ({e}) -> {wait_sec}秒後にリトライ ({attempt}/{max_retries})...")
                time.sleep(wait_sec)
    raise RuntimeError(f"{area} 取得最終失敗 ({max_retries}回試行): {last_err}")

def build_features(hist, wfm):

    per = {}

    for ar in AREAS:

        x = wfm.engineer(hist, ar, 1).set_index("target_datetime")

        if x.index.duplicated().any(): raise RuntimeError(f"{ar}: target_datetime重複")

        per[ar] = x

    idx = sorted(set().union(*[set(x.index) for x in per.values()]))

    f = pd.DataFrame(index=pd.DatetimeIndex(idx, name="target_datetime"))

    for v in NAT_VARS:

        panel = pd.concat({ar: per[ar][v] for ar in AREAS if v in per[ar].columns}, axis=1).reindex(f.index)

        s = v[3:]

        f[f"wf_nat_{s}_mean"] = panel.mean(axis=1); f[f"wf_nat_{s}_min"] = panel.min(axis=1)

        f[f"wf_nat_{s}_max"] = panel.max(axis=1); f[f"wf_nat_{s}_std"] = panel.std(axis=1)

    for ar in AREAS:

        for v in PER_VARS:

            if v in per[ar].columns: f[f"wf_{ar}_{v[3:]}"] = per[ar][v].reindex(f.index)

    return pd.concat([per["tokyo"].reindex(f.index), f], axis=1)

def compare(feat, train_path):

    tr = pd.read_parquet(train_path); tr["target_datetime"] = pd.to_datetime(tr["target_datetime"])

    cols = [c for c in tr.columns if c.startswith("wf_")]

    print(f"学習側のwf_列={len(cols)}  今回の列={feat.shape[1]}  学習のみ={sorted(set(cols) - set(feat.columns))[:5]}  今回のみ={sorted(set(feat.columns) - set(cols))[:5]}")

    t = tr.drop_duplicates("target_datetime").set_index("target_datetime")[[c for c in cols if c in feat.columns]]

    j = feat.index.intersection(t.index); print(f"比較する行={len(j)} ({j.min()} ~ {j.max()})")

    if len(j) == 0: print("共通の行がありません"); return 1

    bad = {}

    for c in t.columns:

        a, b = feat.loc[j, c].to_numpy(float), t.loc[j, c].to_numpy(float)

        n = int((~np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)).sum())

        if n: bad[c] = (n, float(np.nanmax(np.abs(a - b))))

    print(f"値が違う列={len(bad)}/{t.shape[1]}")

    for c, (n, m) in sorted(bad.items(), key=lambda z: -z[1][0])[:8]: print(f"  {c}: {n}行 最大差{m:.4g}")

    return 0 if not bad else 2

def main():

    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["fetch", "replay"], default="fetch")

    ap.add_argument("--out", type=Path, default=DB / "future_weather_features_system_day1.parquet")

    ap.add_argument("--days", type=int, default=8); ap.add_argument("--replay-days", type=int, default=10)

    ap.add_argument("--train", type=Path, default=DB / "training_v8_price_regime" / "jepx_training_v8_price_regime_system_natwf_day1.parquet")

    ap.add_argument("--no-ssl-verify", action="store_true"); a = ap.parse_args()

    bf = load(WDIR / "backfill_open_meteo_forecast_history.py", "bf"); wfm = load(WDIR / "build_weather_forecast_features_day1.py", "wfm")

    if a.mode == "replay":

        h = pd.read_parquet(WDIR / "forecast_history" / "weather_forecast_history_30min.parquet"); h["target_datetime"] = pd.to_datetime(h["target_datetime"])

        h = h[h["forecast_horizon_days"] == 1]; lim = h["target_datetime"].max().normalize()

        h = h[h["target_datetime"] >= lim - pd.Timedelta(days=a.replay_days)]

        feat = build_features(h, wfm); feat = feat[(feat.index >= feat.index.min().normalize() + pd.Timedelta(days=1)) & (feat.index < lim)]

        print(f"replay: 履歴 {h['target_datetime'].min()} ~ {h['target_datetime'].max()}")

        sys.exit(compare(feat, a.train))

    sess = requests.Session(); sess.headers["User-Agent"] = "jepx-weather-future/1.0"; t0 = date.today(); fr = []

    for ar in AREAS:

        print(f"取得中: {ar}"); fr.append(fetch_area(sess, bf, ar, t0, t0 + timedelta(days=a.days), not a.no_ssl_verify))

    hist = bf.to30(pd.concat(fr, ignore_index=True)); feat = build_features(hist, wfm)

    feat = feat[(feat.index >= pd.Timestamp(t0 + timedelta(days=1))) & (feat.index < pd.Timestamp(t0 + timedelta(days=a.days + 1)))]

    nn = feat.isna().sum(); print(f"将来の行={len(feat)} ({feat.index.min()} ~ {feat.index.max()}) wf_列={feat.shape[1]} 欠損のある列={int((nn > 0).sum())}")

    if (nn > 0).any(): print("  欠損の多い列:", nn[nn > 0].sort_values(ascending=False).head(5).to_dict())

    a.out.parent.mkdir(parents=True, exist_ok=True); feat.reset_index().to_parquet(a.out, index=False); print("出力:", a.out)

if __name__ == "__main__": main()

