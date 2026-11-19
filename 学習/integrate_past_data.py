#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9エリア需給実績・過去気象・JEPX・OCCTO時点再現特徴量を datetime/area で統合する。
v3: 需給が他エリアより遅れているエリア(中部の月次更新等)は、全体最新時刻まで需給欠損行を補い、
    気象・JEPX・OCCTOを結合して残す(value_status=no_supply_pending)。
    OCCTO特徴量は既定で pre1100(翌日分・11:00前の予測用)のみ結合する。
    post1100 は公開時刻を18時ルールで作り直すまで既定では結合しない(--with-post1100 で有効化)。
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(r"P:\エリアプライス2\database")
AREA_MAP = {
    "北海道": "hokkaido", "東北": "tohoku", "東京": "tokyo", "中部": "chubu",
    "北陸": "hokuriku", "関西": "kansai", "中国": "chugoku", "四国": "shikoku", "九州": "kyushu"
}
JEPX_COMMON = ["システムプライス(円/kWh)", "売り入札量(kWh)", "買い入札量(kWh)", "約定総量(kWh)"]
OCCTO_FILES = {
    "pre1100": "occto_features_pre1100_dayahead.parquet",
    "post1100": "occto_features_post1100_dayaftertomorrow.parquet",
}
PAD_STATUS = "no_supply_pending"


def read_csv_auto(path):
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(f"文字コードを判定できません: {path}")


def read_table_auto(csv_path: Path) -> pd.DataFrame:
    if csv_path.is_file():
        return read_csv_auto(csv_path)
    pq_path = csv_path.with_suffix(".parquet")
    if pq_path.is_file():
        return pd.read_parquet(pq_path)
    raise FileNotFoundError(csv_path)


def normalize_area(s):
    s = s.astype(str).str.strip().str.lower()
    return s.replace({k.lower(): v for k, v in AREA_MAP.items()})


def assert_unique(df, keys, name):
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(10).to_dict("records")
        raise ValueError(f"{name}のキーが重複しています: {sample}")


def pad_lagging_areas(supply):
    """需給が全体最新時刻より前で止まっているエリアに、欠損行を追加する。"""
    gmax = supply["datetime"].max()
    pads = []
    for area, amax in supply.groupby("area")["datetime"].max().items():
        if amax >= gmax:
            continue
        idx = pd.date_range(amax + pd.Timedelta(minutes=30), gmax, freq="30min")
        if len(idx) == 0:
            continue
        p = pd.DataFrame({"datetime": idx, "area": area})
        if "value_status" in supply.columns:
            p["value_status"] = PAD_STATUS
        pads.append(p)
        print(f"PAD {area}: {idx[0]} .. {idx[-1]} rows={len(idx):,}（需給欠損・気象/JEPX/OCCTOは結合）")
    if not pads:
        print("PAD なし（全エリア需給が最新）")
        return supply
    return pd.concat([supply, *pads], ignore_index=True)


def load_occto_features(path, mode):
    if not path.is_file():
        raise FileNotFoundError(f"OCCTO特徴量がありません: {path}")
    df = pd.read_parquet(path)
    missing = {"datetime", "area"} - set(df.columns)
    if missing:
        raise KeyError(f"{path.name}に必須列がありません: {sorted(missing)}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise").dt.floor("30min")
    df["area"] = normalize_area(df["area"])
    df = df[df["area"].isin(AREA_MAP.values())].copy()
    assert_unique(df, ["datetime", "area"], path.name)
    # 2モデルの特徴量を同時収録できるよう、モード接頭辞を付ける。
    rename = {c: f"{mode}_{c}" for c in df.columns if c not in {"datetime", "area"}}
    return df.rename(columns=rename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out-dir", type=Path, default=None, help="出力先。省略時は 統合データ")
    ap.add_argument("--with-post1100", action="store_true", help="post1100のOCCTO特徴量も結合(公開時刻の作り直し後に使用)")
    ap.add_argument("--no-csv", action="store_true", help="CSVを出力しない(検証用)")
    a = ap.parse_args()
    root = a.root
    out_dir = a.out_dir or (root / "統合データ")
    occto_dir = root / "occtoデータ" / "features"
    modes = {k: v for k, v in OCCTO_FILES.items() if a.with_post1100 or k == "pre1100"}

    supply_path = root / "送配電９社データ" / "raw_9area" / "area_supply_master_v2.csv"
    weather_path = root / "天気9エリア" / "data" / "weather_history_9area_30min.csv"
    jepx_path = root / "JPEXデータ" / "data" / "jepx_spot_summary_master.csv"
    required_paths = [p if p.is_file() else p.with_suffix(".parquet") for p in (supply_path, weather_path, jepx_path)] + [occto_dir / f for f in modes.values()]
    for p in required_paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    supply = read_table_auto(supply_path)
    weather = read_table_auto(weather_path)
    jepx = read_table_auto(jepx_path)

    for name, df in (("需給", supply), ("気象", weather)):
        if not {"datetime", "area"}.issubset(df.columns):
            raise KeyError(f"{name}に datetime/area がありません")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="raise").dt.floor("30min")
        df["area"] = normalize_area(df["area"])
        assert_unique(df, ["datetime", "area"], name)

    supply = pad_lagging_areas(supply)

    required = {"受渡日", "時刻コード", *JEPX_COMMON}
    missing = required - set(jepx.columns)
    if missing:
        raise KeyError(f"JEPX必須列不足: {sorted(missing)}")
    code_num = pd.to_numeric(jepx["時刻コード"], errors="raise")
    if not code_num.between(1, 48).all():
        raise ValueError("JEPX時刻コードに1～48以外があります")
    jepx["datetime"] = pd.to_datetime(jepx["受渡日"], errors="raise") + pd.to_timedelta((code_num - 1) * 30, unit="m")

    common = jepx[["datetime", *JEPX_COMMON]].drop_duplicates("datetime")
    parts = []
    for jp_area, en_area in AREA_MAP.items():
        price_col = f"エリアプライス{jp_area}(円/kWh)"
        if price_col not in jepx.columns:
            raise KeyError(f"JEPX列不足: {price_col}")
        x = jepx[["datetime", price_col]].rename(columns={price_col: "area_price_yen_kwh"}).copy()
        x["area"] = en_area
        parts.append(x)
    jepx_long = pd.concat(parts, ignore_index=True).merge(common, on="datetime", how="left")
    jepx_long = jepx_long.drop_duplicates(["datetime", "area"])
    assert_unique(jepx_long, ["datetime", "area"], "JEPX")

    result = supply.merge(weather, on=["datetime", "area"], how="left", suffixes=("", "_weather"))
    result = result.merge(jepx_long, on=["datetime", "area"], how="left")

    occto_stats = []
    for mode, filename in modes.items():
        occto = load_occto_features(occto_dir / filename, mode)
        before = len(result)
        result = result.merge(occto, on=["datetime", "area"], how="left", validate="one_to_one")
        if len(result) != before:
            raise RuntimeError(f"{mode} OCCTO結合で行数が変化しました: {before} -> {len(result)}")
        pc = f"{mode}_prediction_datetime"
        matched = int(result[pc].notna().sum()) if pc in result else 0
        mc = f"{mode}_occto_missing_flag"
        unavailable = int(result[mc].eq(1).sum()) if mc in result else 0
        vc = f"{mode}_occto_area_reserve_mw"
        last = result.loc[result[vc].notna(), "datetime"].max() if vc in result else None
        occto_stats.append((mode, matched, unavailable, last))

    result = result.sort_values(["datetime", "area"]).drop_duplicates(["datetime", "area"], keep="last")
    assert_unique(result, ["datetime", "area"], "統合結果")

    jepx_matched = int(result["システムプライス(円/kWh)"].notna().sum())
    overlap_expected = int(result["datetime"].between(jepx_long["datetime"].min(), jepx_long["datetime"].max()).sum())
    if overlap_expected and jepx_matched == 0:
        raise RuntimeError("JEPX対象期間と重なる需給行があるのにJEPX結合件数が0です")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "past_integrated_9area_30min.csv"
    pq_path = out_dir / "past_integrated_9area_30min.parquet"
    if not a.no_csv:
        result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    result.to_parquet(pq_path, index=False)

    print(f"SUCCESS rows={len(result):,} cols={len(result.columns)} JEPX_matched={jepx_matched:,} JEPX_missing={len(result)-jepx_matched:,}")
    for mode, matched, unavailable, last in occto_stats:
        print(f"OCCTO_{mode}_matched={matched:,} asof_missing={unavailable:,} area_reserve値あり最終={last}")
    print(f"period={result['datetime'].min()} .. {result['datetime'].max()}")
    print(f"CSV={'(出力せず)' if a.no_csv else csv_path}")
    print(f"PARQUET={pq_path}")


if __name__ == "__main__":
    main()
