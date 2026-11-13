# -*- coding: utf-8 -*-
r"""P:\エリアプライス2\web 固定。予測履歴化＋JEPX実績結合（実績のみの行も保持）。"""
from pathlib import Path
import io, time, urllib.request, urllib.parse
import pandas as pd

WEB = Path(r"P:\エリアプライス2\web")
LATEST_CANDIDATES = [
    WEB / "forecast_day1_to_day7_latest.csv",
    Path(r"P:\エリアプライス2\予測\forecast_day1_to_day7_latest.csv"),
    Path(r"P:\エリアプライス2\学習\forecasts\forecast_day1_to_day7_latest.csv"),
    Path(r"P:\エリアプライス2\学習\forecast_day1_to_day7_latest.csv"),
]
HISTORY = WEB / "forecast_day1_to_day7_history.csv"
ACCURACY = WEB / "forecast_day1_to_day7_accuracy.csv"
CACHE = WEB / "jepx_actual_cache"
MASTER_JEPX = Path(r"P:\エリアプライス2\database\JPEXデータ\data\jepx_spot_summary_master.csv")

AREA = {
    "system": "システムプライス", "hokkaido": "エリアプライス北海道", "tohoku": "エリアプライス東北",
    "tokyo": "エリアプライス東京", "chubu": "エリアプライス中部", "hokuriku": "エリアプライス北陸",
    "kansai": "エリアプライス関西", "chugoku": "エリアプライス中国", "shikoku": "エリアプライス四国",
    "kyushu": "エリアプライス九州"
}
AREA_JA = {
    "system": "システム", "hokkaido": "北海道", "tohoku": "東北",
    "tokyo": "東京", "chubu": "中部", "hokuriku": "北陸",
    "kansai": "関西", "chugoku": "中国", "shikoku": "四国", "kyushu": "九州"
}


def latest():
    for p in LATEST_CANDIDATES:
        if p.exists():
            return p
    return None


def fy(d):
    return d.year if d.month >= 4 else d.year - 1


def download(y):
    q = urllib.parse.urlencode({"dir": "spot_summary", "file": f"spot_summary_{y}.csv"}).encode()
    req = urllib.request.Request(
        "https://www.jepx.jp/_download.php?timestamp=" + str(int(time.time() * 1000)),
        data=q,
        headers={
            "Referer": "https://www.jepx.jp/electricpower/market-data/spot/",
            "User-Agent": "Mozilla/5.0"
        }
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    for e in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=e, low_memory=False)
        except Exception:
            pass
    raise RuntimeError("JEPX CSV読込失敗")


def pick(cols, parts):
    return next((c for c in cols if all(p in str(c).replace(" ", "") for p in parts)), None)


def actual(raw):
    dc = pick(raw.columns, ["受渡日"]) or pick(raw.columns, ["年月日"])
    sc = pick(raw.columns, ["時刻コード"])
    if not dc or not sc:
        raise ValueError("JEPX列不足")
    z = []
    for a, label in AREA.items():
        c = pick(raw.columns, [label, "円/kWh"])
        if c:
            q = pd.DataFrame({
                "target_date": pd.to_datetime(raw[dc], errors="coerce").dt.strftime("%Y-%m-%d"),
                "slot": pd.to_numeric(raw[sc], errors="coerce"),
                "area": a,
                "area_ja": AREA_JA[a],
                "actual_price_yen_kwh": pd.to_numeric(raw[c], errors="coerce")
            })
            z.append(q)
    return pd.concat(z, ignore_index=True).dropna(subset=["target_date", "slot", "actual_price_yen_kwh"])


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    lp = latest()
    old = pd.read_csv(HISTORY, encoding="utf-8-sig", low_memory=False) if HISTORY.exists() else pd.DataFrame()

    if lp:
        x = pd.read_csv(lp, encoding="utf-8-sig", low_memory=False)
        dt = pd.to_datetime(x["target_datetime"], errors="coerce")
        x["target_date"] = dt.dt.strftime("%Y-%m-%d")
        x["slot"] = (dt.dt.hour * 2 + dt.dt.minute // 30 + 1).astype(int)
        hist = pd.concat([old, x], ignore_index=True).drop_duplicates(["forecast_created_at", "target_datetime", "area"], keep="last")
    else:
        hist = old

    # target_date と slot の欠損を target_datetime から確実に補完
    if not hist.empty and "target_datetime" in hist.columns:
        dt = pd.to_datetime(hist["target_datetime"], errors="coerce")
        if "target_date" not in hist.columns:
            hist["target_date"] = dt.dt.strftime("%Y-%m-%d")
        else:
            hist["target_date"] = hist["target_date"].fillna(dt.dt.strftime("%Y-%m-%d"))
        if "slot" not in hist.columns:
            hist["slot"] = (dt.dt.hour * 2 + dt.dt.minute // 30 + 1).fillna(1).astype(int)
        else:
            hist["slot"] = hist["slot"].fillna((dt.dt.hour * 2 + dt.dt.minute // 30 + 1)).fillna(1).astype(int)

    hist.to_csv(HISTORY, index=False, encoding="utf-8-sig")

    # 年度 (fiscal year) の安全な抽出 (NaT/NaN を完全に除外)
    today = pd.Timestamp.now().date()
    years = {fy(today)}
    if not hist.empty and "target_date" in hist.columns:
        valid_dates = pd.to_datetime(hist["target_date"], errors="coerce").dropna().dt.date
        years |= {fy(d) for d in valid_dates}
    valid_years = sorted({int(y) for y in years if pd.notna(y) and str(y).replace(".0", "").isdigit()})
    print(f"Target fiscal years: {valid_years}")

    acts = []
    for y in valid_years:
        p = CACHE / f"spot_summary_{y}.csv"
        r = None
        try:
            r = download(y)
            r.to_csv(p, index=False, encoding="utf-8-sig")
        except Exception as e:
            if p.exists():
                print(f"WARN cache reused for {y}: {e}")
                r = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
            elif MASTER_JEPX.exists():
                print(f"WARN fallback to JEPX master for {y}: {e}")
                r = pd.read_csv(MASTER_JEPX, encoding="utf-8-sig", low_memory=False)
            else:
                raise
        acts.append(actual(r))

    a = pd.concat(acts, ignore_index=True).drop_duplicates(["target_date", "slot", "area"], keep="last")

    # 実績と予測のマージ
    if hist.empty:
        z = a.copy()
    else:
        z = hist.merge(a, on=["target_date", "slot", "area"], how="outer", suffixes=("", "_actual"))

    if "area_ja_actual" in z:
        z["area_ja"] = z.get("area_ja").fillna(z["area_ja_actual"])
        z = z.drop(columns=["area_ja_actual"])

    z["predicted_price_yen_kwh"] = pd.to_numeric(z.get("predicted_price_yen_kwh"), errors="coerce")
    z["actual_price_yen_kwh"] = pd.to_numeric(z["actual_price_yen_kwh"], errors="coerce")
    z["error"] = z["predicted_price_yen_kwh"] - z["actual_price_yen_kwh"]
    z["abs_error"] = z["error"].abs()

    z.to_csv(ACCURACY, index=False, encoding="utf-8-sig")
    print(f"SUCCESS history={len(hist):,} accuracy={len(z):,} actual={z.actual_price_yen_kwh.notna().sum():,}")


if __name__ == "__main__":
    main()
