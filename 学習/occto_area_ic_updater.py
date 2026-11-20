#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""エリア別OCCTO予備率・需給および連系線空容量特徴量 (occto_area_ic_da.parquet) の更新＆共通読込モジュール。

- bulk_downloads (月次CSV) と downloads (日次翌々日・翌日CSV) の両方を統合し、
  既存 occto_area_ic_da.parquet (148列) と完全互換のテーブルを最新日+DAY1分まで維持する。
- 8エリアモデル (学習/推論) から共通で呼び出される select_area_occto_cols / load_occto_merged を提供。
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

BASE_OCCTO = Path(r"P:\エリアプライス2\database\occtoデータ")
DA_DIR = Path(r"P:\エリアプライス2\学習\evaluation_system_day1_levers_da")
DA8_PATH = DA_DIR / "occto_da8.parquet"
AIC_PATH = DA_DIR / "occto_area_ic_da.parquet"

AREA_JP_TO_PREF = {
    "北海道": "hok",
    "東北": "toh",
    "東京": "tok",
    "中部": "chu",
    "北陸": "hkr",
    "関西": "kan",
    "中国": "chg",
    "四国": "shk",
    "九州": "kyu",
}

AREA_EN_TO_PREF = {
    "hokkaido": "hok",
    "tohoku": "toh",
    "tokyo": "tok",
    "chubu": "chu",
    "hokuriku": "hkr",
    "kansai": "kan",
    "chugoku": "chg",
    "shikoku": "shk",
    "kyushu": "kyu",
}

IC_KEY = {
    "北海道・本州間電力連系設備": "hokuhon",
    "相馬双葉幹線": "soma",
    "周波数変換設備": "fc",
    "関西-中国（東）": "kc_e",
    "関西-中国（西）": "kc_w",
    "関門連系線": "kanmon",
}

IC_DETAIL_KEYS = ["fc", "soma", "hokuhon", "kc_e", "kanmon"]


def _read_csv_occto(p: Path) -> pd.DataFrame | None:
    if not p.is_file() or p.stat().st_size == 0:
        return None
    for enc in ("utf-8-sig", "cp932"):
        try:
            with p.open("r", encoding=enc) as f:
                first = f.readline()
            sk = 1 if "UPDATE" in first else 0
            df = pd.read_csv(p, encoding=enc, skiprows=sk, low_memory=False)
            df.columns = [str(c).strip().rstrip(".") for c in df.columns]
            if len(df) == 0:
                return None
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def _parse_dt(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    t = time_s.astype(str).str.strip().str.replace(r"^(\d):", r"0\1:", regex=True)
    hh = pd.to_numeric(t.str[:2], errors="coerce")
    mm = pd.to_numeric(t.str[3:5], errors="coerce")
    base = pd.to_datetime(date_s.astype(str).str.strip().str.replace("-", "/"), errors="coerce")
    return base + pd.to_timedelta(hh * 60 + mm - 30, unit="m")


def build_area_wide_df(base: Path = BASE_OCCTO, include_downloads: bool = True) -> pd.DataFrame:
    """bulk_downloads と downloads から 9エリア×10列 + 東西集計16列 (計106列) を構築する。"""
    fs = []
    if include_downloads:
        fs += sorted((base / "downloads").glob("occto_area_wide_block_reserve_dayaftertomorrow_*.csv"), key=lambda p: p.stat().st_mtime)
    fs += sorted((base / "bulk_downloads" / "wide_block_dayaftertomorrow").glob("*.csv"), key=lambda p: p.stat().st_mtime)

    parts = []
    for p in fs:
        d = _read_csv_occto(p)
        if d is None:
            continue
        acol = "エリア名" if "エリア名" in d.columns else ("エリア" if "エリア" in d.columns else None)
        dcol = "対象年月日" if "対象年月日" in d.columns else ("年月日" if "年月日" in d.columns else None)
        if not acol or not dcol or "時刻" not in d.columns:
            continue
        d = d[d[acol].isin(AREA_JP_TO_PREF)].copy()
        if d.empty:
            continue
        d["dt"] = _parse_dt(d[dcol], d["時刻"])
        d = d.dropna(subset=["dt"])
        d["_pref"] = d[acol].map(AREA_JP_TO_PREF)
        col_map = {
            "エリア需要(MW)": "area_demand",
            "エリア供給力(MW)": "area_supply",
            "エリア予備力(MW)": "area_reserve",
            "広域予備率(%)": "wide_res_rate",
            "広域使用率(%)": "wide_util",
            "広域ブロック需要(MW)": "blk_demand",
            "広域ブロック供給力(MW)": "blk_supply",
            "広域ブロック予備力(MW)": "blk_reserve",
        }
        keep = ["dt", "_pref"]
        for jp, en in col_map.items():
            if jp in d.columns:
                d[en] = pd.to_numeric(d[jp], errors="coerce")
                keep.append(en)
        parts.append(d[keep])

    if not parts:
        return pd.DataFrame()

    all_d = pd.concat(parts, ignore_index=True).drop_duplicates(["dt", "_pref"], keep="last")
    all_d["area_util"] = 100.0 * all_d["area_demand"] / all_d["area_supply"].replace(0, np.nan)
    all_d["area_res_rate"] = 100.0 * all_d["area_reserve"] / all_d["area_demand"].replace(0, np.nan)

    metrics = [
        "area_demand", "area_supply", "area_reserve", "area_util", "area_res_rate",
        "wide_res_rate", "wide_util", "blk_demand", "blk_supply", "blk_reserve",
    ]
    piv = all_d.pivot(index="dt", columns="_pref", values=[m for m in metrics if m in all_d.columns])
    out = pd.DataFrame(index=piv.index)
    for pref in AREA_JP_TO_PREF.values():
        for m in metrics:
            if (m, pref) in piv.columns:
                out[f"occ_{pref}_{m}"] = piv[(m, pref)]

    # 東日本50Hz本州 (toh, tok) / 西日本60Hz (chu, hkr, kan, chg, shk, kyu) 集計・差分 (16列)
    e_prefs = ["toh", "tok"]
    w_prefs = ["chu", "hkr", "kan", "chg", "shk", "kyu"]
    for grp_name, prefs in (("east50", e_prefs), ("west60", w_prefs)):
        for m_src, m_dst in (("area_demand", "demand"), ("area_supply", "supply"), ("area_reserve", "reserve")):
            cols = [f"occ_{p}_{m_src}" for p in prefs if f"occ_{p}_{m_src}" in out.columns]
            if cols:
                out[f"occ_{grp_name}_{m_dst}"] = out[cols].sum(axis=1, min_count=len(cols))
        if f"occ_{grp_name}_demand" in out.columns and f"occ_{grp_name}_supply" in out.columns:
            out[f"occ_{grp_name}_util"] = 100.0 * out[f"occ_{grp_name}_demand"] / out[f"occ_{grp_name}_supply"].replace(0, np.nan)
        if f"occ_{grp_name}_reserve" in out.columns and f"occ_{grp_name}_demand" in out.columns:
            out[f"occ_{grp_name}_res_rate"] = 100.0 * out[f"occ_{grp_name}_reserve"] / out[f"occ_{grp_name}_demand"].replace(0, np.nan)

    if {"occ_tok_area_util", "occ_chu_area_util"}.issubset(out.columns):
        out["occ_diff_tok_chu_util"] = out["occ_tok_area_util"] - out["occ_chu_area_util"]
    if {"occ_tok_area_util", "occ_toh_area_util"}.issubset(out.columns):
        out["occ_diff_tok_toh_util"] = out["occ_tok_area_util"] - out["occ_toh_area_util"]
    if {"occ_east50_util", "occ_west60_util"}.issubset(out.columns):
        out["occ_diff_east_west_util"] = out["occ_east50_util"] - out["occ_west60_util"]
    if {"occ_east50_res_rate", "occ_west60_res_rate"}.issubset(out.columns):
        out["occ_diff_east_west_res_rate"] = out["occ_east50_res_rate"] - out["occ_west60_res_rate"]
    if {"occ_tok_wide_res_rate", "occ_chu_wide_res_rate"}.issubset(out.columns):
        out["occ_diff_tok_chu_wide_res"] = out["occ_tok_wide_res_rate"] - out["occ_chu_wide_res_rate"]
    if {"occ_tok_wide_util", "occ_chu_wide_util"}.issubset(out.columns):
        out["occ_diff_tok_chu_wide_util"] = out["occ_tok_wide_util"] - out["occ_chu_wide_util"]

    out.index.name = "target_datetime"
    return out.sort_index()


def _normalize_ic_wide(d: pd.DataFrame) -> pd.DataFrame | None:
    """bulk_downloads (横持ち) または downloads (縦持ち) の連系線CSVを共通横持ち形式へ正規化する。"""
    if "順方向空容量(MW)" in d.columns and "連系線名" in d.columns:
        dcol = "対象年月日" if "対象年月日" in d.columns else "年月日"
        out = pd.DataFrame({
            "dt": _parse_dt(d[dcol], d["時刻"]),
            "line": d["連系線名"].astype(str).str.strip(),
            "fwd_cap": pd.to_numeric(d["順方向運用容量(MW)"], errors="coerce"),
            "bwd_cap": pd.to_numeric(d["逆方向運用容量(MW)"], errors="coerce"),
            "fwd_margin": pd.to_numeric(d["順方向マージン(MW)"], errors="coerce"),
            "bwd_margin": pd.to_numeric(d["逆方向マージン(MW)"], errors="coerce"),
            "fwd_flow": pd.to_numeric(d["順方向予想潮流(MW)"], errors="coerce"),
            "bwd_flow": pd.to_numeric(d["逆方向予想潮流(MW)"], errors="coerce"),
            "fwd_free": pd.to_numeric(d["順方向空容量(MW)"], errors="coerce"),
            "bwd_free": pd.to_numeric(d["逆方向空容量(MW)"], errors="coerce"),
        })
        return out.dropna(subset=["dt"])

    if "方向" in d.columns and "連系線" in d.columns and "空容量" in d.columns:
        dcol = "年月日" if "年月日" in d.columns else "対象年月日"
        d = d.copy()
        d["dt"] = _parse_dt(d[dcol], d["時刻"])
        d["line"] = d["連系線"].astype(str).str.strip()
        d["dir"] = d["方向"].astype(str).str.strip()
        for c in ("空容量", "計画潮流", "マージン", "運用容量"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["dt"]).drop_duplicates(["dt", "line", "dir"], keep="last")
        fwd = d[d["dir"] == "順方向"].set_index(["dt", "line"])
        bwd = d[d["dir"] == "逆方向"].set_index(["dt", "line"])
        idx = fwd.index.union(bwd.index)
        if len(idx) == 0:
            return None
        # 縦持ちCSVの逆方向は負号表記のため、横持ち仕様と整合させる (cap/marginは絶対値、freeは-bwd_val、flowは-fwd_flow相当)
        out = pd.DataFrame(index=idx)
        out["fwd_cap"] = fwd["運用容量"].reindex(idx).abs()
        out["bwd_cap"] = bwd["運用容量"].reindex(idx).abs()
        out["fwd_margin"] = fwd["マージン"].reindex(idx).abs()
        out["bwd_margin"] = bwd["マージン"].reindex(idx).abs()
        out["fwd_flow"] = fwd["計画潮流"].reindex(idx)
        out["bwd_flow"] = bwd["計画潮流"].reindex(idx)
        # 逆方向空容量: 縦持ちでは負号で表記される (例: -50 => 50MW相当。順方向と逆符号)
        out["fwd_free"] = fwd["空容量"].reindex(idx)
        out["bwd_free"] = -bwd["空容量"].reindex(idx)
        return out.reset_index()

    return None


def build_ic_df(base: Path = BASE_OCCTO, include_downloads: bool = True) -> pd.DataFrame:
    """連系線CSVから 57列の ic_* 特徴量を構築する (bulk_downloads優先、最新日のみdownloads補完)。"""
    fs = []
    if include_downloads:
        fs += sorted((base / "downloads").glob("occto_interconnection_capacity_tomorrow_*.csv"), key=lambda p: p.stat().st_mtime)
        fs += sorted((base / "downloads").glob("occto_interconnection_capacity_dayaftertomorrow_*.csv"), key=lambda p: p.stat().st_mtime)
    fs += sorted((base / "bulk_downloads" / "interconnection_dayaftertomorrow").glob("*.csv"), key=lambda p: p.stat().st_mtime)

    parts = []
    for p in fs:
        raw = _read_csv_occto(p)
        if raw is None:
            continue
        norm = _normalize_ic_wide(raw)
        if norm is not None and not norm.empty:
            parts.append(norm)

    if not parts:
        return pd.DataFrame()

    d = pd.concat(parts, ignore_index=True).drop_duplicates(["dt", "line"], keep="last")
    d["abs_flow"] = d["fwd_flow"].abs()
    d["cap"] = d["fwd_cap"] + d["bwd_cap"]
    d["margin"] = d["fwd_margin"] + d["bwd_margin"]

    g = d.groupby("dt")
    f = pd.DataFrame({
        "ic_fwd_free_min": g["fwd_free"].min(),
        "ic_bwd_free_min": g["bwd_free"].min(),
        "ic_fwd_free_sum": g["fwd_free"].sum(min_count=1),
        "ic_bwd_free_sum": g["bwd_free"].sum(min_count=1),
        "ic_cap_sum": g["cap"].sum(min_count=1),
        "ic_margin_sum": g["margin"].sum(min_count=1),
        "ic_flow_abs_max": g["abs_flow"].max(),
        "ic_count": g.size().astype(float),
    })
    f["ic_free_ratio_min"] = (
        d.assign(r=(d["fwd_free"].clip(lower=0) + d["bwd_free"].clip(lower=0)) / d["cap"].replace(0, np.nan))
        .groupby("dt")["r"].min()
    )

    key_dfs = {}
    for name, k in IC_KEY.items():
        s = d[d["line"] == name].set_index("dt")
        key_dfs[k] = s
        f[f"ic_{k}_fwd_free"] = s["fwd_free"]
        f[f"ic_{k}_bwd_free"] = s["bwd_free"]

    for k in IC_DETAIL_KEYS:
        s = key_dfs[k]
        f[f"ic_{k}_fwd_flow"] = s["fwd_flow"]
        f[f"ic_{k}_bwd_flow"] = s["bwd_flow"]
        f[f"ic_{k}_fwd_cap"] = s["fwd_cap"]
        f[f"ic_{k}_bwd_cap"] = s["bwd_cap"]
        f[f"ic_{k}_fwd_free_ratio"] = s["fwd_free"] / s["fwd_cap"].replace(0, np.nan)
        f[f"ic_{k}_bwd_free_ratio"] = s["bwd_free"] / s["bwd_cap"].replace(0, np.nan)
        f[f"ic_{k}_min_free"] = np.minimum(s["fwd_free"], s["bwd_free"])

    f.index.name = "target_datetime"
    return f.sort_index()


def update_occto_area_ic(base: Path = BASE_OCCTO, out_path: Path = AIC_PATH) -> pd.DataFrame:
    """既存の occto_area_ic_da.parquet に最新 bulk_downloads / downloads 分を差分マージして保存する。"""
    old = None
    if out_path.is_file():
        old = pd.read_parquet(out_path)
        old["target_datetime"] = pd.to_datetime(old["target_datetime"])
        old = old.drop_duplicates("target_datetime", keep="last").set_index("target_datetime").sort_index()

    area_df = build_area_wide_df(base, include_downloads=True)
    ic_df = build_ic_df(base, include_downloads=True)

    if area_df.empty and ic_df.empty:
        print("WARN occto_area_ic: 新規CSVが見つかりません。既存ファイルを維持します。")
        return old if old is not None else pd.DataFrame()

    if not area_df.empty and not ic_df.empty:
        new = area_df.join(ic_df, how="outer")
    else:
        new = area_df if not area_df.empty else ic_df

    if old is not None:
        all_cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
        # 既存履歴を優先保持し、新規日付・欠損セルのみ new で補完 (既存期間の数値破壊を防止)
        old_aligned = old.reindex(index=old.index.union(new.index), columns=all_cols)
        new_aligned = new.reindex(index=old_aligned.index, columns=all_cols)
        merged = old_aligned.combine_first(new_aligned).sort_index()
        old_max = old.index.max()
        old_rows = len(old)
    else:
        merged = new.sort_index()
        old_max = None
        old_rows = 0

    out_df = merged.reset_index().rename(columns={"index": "target_datetime"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        shutil.copy2(out_path, out_path.with_name(out_path.stem + ".prev.parquet"))
    out_df.to_parquet(out_path, index=False)
    print(
        f"OK occto_area_ic_da: {len(out_df):,}行 {merged.index.min()}〜{merged.index.max()} "
        f"(前回 {old_rows:,}行 〜{old_max})"
    )
    return merged


def load_occto_merged(base: Path = BASE_OCCTO, da8_path: Path = DA8_PATH, aic_path: Path = AIC_PATH, include_live_downloads: bool = False) -> pd.DataFrame:
    """occto_area_ic_da + occto_da8 を target_datetime インデックスで統合して返す。"""
    dfs = []
    if aic_path.is_file():
        aic = pd.read_parquet(aic_path)
        aic["target_datetime"] = pd.to_datetime(aic["target_datetime"])
        dfs.append(aic.drop_duplicates("target_datetime", keep="last").set_index("target_datetime"))
    if da8_path.is_file():
        da8 = pd.read_parquet(da8_path)
        da8["target_datetime"] = pd.to_datetime(da8["target_datetime"])
        dfs.append(da8.drop_duplicates("target_datetime", keep="last").set_index("target_datetime"))
    if not dfs:
        return pd.DataFrame()

    occ = dfs[0]
    for d in dfs[1:]:
        new_c = [c for c in d.columns if c not in occ.columns]
        if new_c:
            occ = occ.join(d[new_c], how="outer")

    if include_live_downloads:
        area_dl = build_area_wide_df(base, include_downloads=True)
        ic_dl = build_ic_df(base, include_downloads=True)
        live = area_dl.join(ic_dl, how="outer") if (not area_dl.empty and not ic_dl.empty) else (area_dl if not area_dl.empty else ic_dl)
        if not live.empty:
            # downloads にある直近日付だけ追加・上書き
            all_cols = list(occ.columns) + [c for c in live.columns if c not in occ.columns]
            occ = live.reindex(index=occ.index.union(live.index), columns=all_cols).combine_first(
                occ.reindex(index=occ.index.union(live.index), columns=all_cols)
            )
            # da8 相当の8列も欠損なら area_dl の9エリア平均で補完
            pref_list = list(AREA_JP_TO_PREF.values())
            avg_map = [
                ("area_demand", "pre1100_occto_area_demand_mw"),
                ("area_supply", "pre1100_occto_area_supply_mw"),
                ("area_reserve", "pre1100_occto_area_reserve_mw"),
                ("blk_demand", "pre1100_occto_block_demand_mw"),
                ("blk_supply", "pre1100_occto_block_supply_mw"),
                ("blk_reserve", "pre1100_occto_block_reserve_mw"),
                ("wide_res_rate", "pre1100_occto_wide_reserve_rate_pct"),
                ("wide_util", "pre1100_occto_wide_utilization_pct"),
            ]
            for src_m, dst_c in avg_map:
                src_cols = [f"occ_{pr}_{src_m}" for pr in pref_list if f"occ_{pr}_{src_m}" in occ.columns]
                if src_cols:
                    avg_s = occ[src_cols].mean(axis=1)
                    if dst_c not in occ.columns:
                        occ[dst_c] = avg_s
                    else:
                        occ[dst_c] = occ[dst_c].fillna(avg_s)

    occ.index.name = "target_datetime"
    return occ.sort_index()


def select_area_occto_cols(occ_cols: list[str], area: str) -> list[str]:
    """8エリアモデルで180日検証済みのOCCTO・連系線特徴量リストを返す。"""
    pref = AREA_EN_TO_PREF[area]
    keep = []
    for c in occ_cols:
        if c.startswith(f"occ_{pref}_"):
            keep.append(c)
        elif c.startswith("pre1100_occto_"):
            keep.append(c)
        elif c.startswith("ic_"):
            keep.append(c)
        elif c.endswith("_area_util") or c.endswith("_area_res_rate") or c.endswith("_wide_res_rate"):
            keep.append(c)
    return sorted(set(keep))


if __name__ == "__main__":
    update_occto_area_ic()
