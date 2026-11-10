#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P:\\エリアプライス2 のクラウド自動実行（GitHub Actions + Power Automate）用パッケージを作成するスクリプト。

作成されるもの:
1. P:\\エリアプライス2\\cloud_release_bundle.zip
   - GitHub Releases (タグ: latest-data) にブラウザからドラッグ＆ドロップで1回アップロードするデータ＋スクリプト一式
   - GitHub Actions 実行後にもこのスクリプトが呼ばれ、最新化されたマスタデータを次回実行用に自動保存する。
"""
import zipfile
from pathlib import Path

ROOT = Path(r"P:\エリアプライス2")
OUT_ZIP = ROOT / "cloud_release_bundle.zip"

# 1. コード・設定ファイルとして含めるディレクトリと拡張子
CODE_DIRS = ["学習", "web", "database", "予測", "仕様", ".github"]
CODE_EXTS = {".py", ".ps1", ".json", ".yml", ".yaml", ".html", ".md", ".txt"}

# 除外するサブフォルダ名（巨大な過去バックアップや一時キャッシュ）
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    "fast_eval_cache",
    "bulk_downloads",
    "logs",
    ".git",
    "cloud_upload_repo",
}

# 2. データファイル（.csv / .parquet）として明示的に含める必須ファイル・パターン
ESSENTIAL_DATA_GLOBS = [
    # 送配電9社データ
    "database/送配電９社データ/source_registry_v4_2.json",
    "database/送配電９社データ/update_area_supply_master_report.json",
    "database/送配電９社データ/raw_9area/area_supply_master_v2.parquet",
    "database/送配電９社データ/raw_9area/area_supply_master_v2.csv",
    "database/送配電９社データ/raw_9area/*2026*.csv",
    # JEPXデータ
    "database/JPEXデータ/data/jepx_spot_summary_master.csv",
    "database/JPEXデータ/data/spot_summary_2026.csv",
    "database/JPEXデータ/data/spot_summary_2025.csv",
    # 天気9エリア
    "database/天気9エリア/data/*.csv",
    "database/天気9エリア/data/*.parquet",
    "database/天気9エリア/*.parquet",
    "database/天気9エリア/*.csv",
    "database/天気9エリア/forecast_history/*.parquet",
    "database/天気9エリア/forecast_history/*.csv",
    # OCCTOデータ (特徴量parquetと直近ダウンロードCSV)
    "database/occtoデータ/features/*.parquet",
    # 発電所停止データ (Outages)
    "database/outages_features_day1.parquet",
    "database/outages_*.csv",
    # 統合データ (日次パイプラインが読み書きするコアファイル)
    "database/統合データ/past_integrated_9area_30min.parquet",
    "database/統合データ/future_integrated_9area_30min.parquet",
    "database/統合データ/future_weather_features_system_day1.parquet",
    "database/統合データ/future_price_features_system_day1.parquet",
    "database/統合データ/future_area_day1_area.parquet",
    "database/統合データ/future_area_day1_premium.parquet",
    "database/統合データ/weather_archive/*.parquet",
    "database/統合データ/training_v8_price_regime/jepx_training_v8_price_regime_system_natwf_day1.parquet",
    "database/統合データ/training_v8_price_regime/jepx_training_v8_price_regime_area_day1.parquet",
    "database/統合データ/training_v8_price_regime/jepx_training_v8_price_regime_day2_to_day7.parquet",
    "database/統合データ/training_v8_price_regime/jepx_training_v8_price_regime_*_day1.parquet",
    "database/統合データ/training_v8_price_regime/jepx_training_v8_*_premium_day1.parquet",
    # 学習フォルダ内の必須データ
    "学習/evaluation_system_day1_levers_da/occto_da8.parquet",
    "学習/evaluation_system_day1_levers_da/occto_area_ic_da.parquet",
    "学習/system_day1_v3_calib.json",
    "学習/tokyo_day1_v4_calib.json",
    # 予測・web出力の最新CSV
    "予測/*latest*.csv",
    "予測/*compare*.csv",
    "web/*.csv",
    "web/*.html",
]


def collect_files() -> list[Path]:
    selected: set[Path] = set()

    # (A) スクリプト・設定ファイル類を収集
    for dname in CODE_DIRS:
        base = ROOT / dname
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in p.relative_to(ROOT).parts):
                continue
            if p.suffix.lower() in CODE_EXTS:
                if p.stat().st_size <= 10 * 1024 * 1024:
                    selected.add(p)

    # (B) 必須データファイルを収集
    for pattern in ESSENTIAL_DATA_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file():
                selected.add(p)

    # (C) OCCTO downloads のうち直近更新された100ファイルのみ追加
    occto_dl = ROOT / "database/occtoデータ/downloads"
    if occto_dl.exists():
        dl_files = sorted(
            (p for p in occto_dl.glob("*.csv") if p.is_file()),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )[:100]
        selected.update(dl_files)

    return sorted(selected)


def main():
    files = collect_files()
    total_raw_bytes = sum(p.stat().st_size for p in files)
    print(f"対象ファイル数: {len(files)} 件 (非圧縮合計: {total_raw_bytes / (1024*1024):.1f} MB)", flush=True)

    tmp_zip = ROOT / "cloud_release_bundle.tmp.zip"
    if tmp_zip.exists():
        tmp_zip.unlink()

    with zipfile.ZipFile(tmp_zip, "w") as zf:
        for idx, p in enumerate(files, 1):
            rel = p.relative_to(ROOT).as_posix()
            # Parquetは既に圧縮済みのため無圧縮格納(高速)、CSV/テキストは高速Deflate圧縮
            if p.suffix.lower() == ".parquet":
                zf.write(p, arcname=rel, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(p, arcname=rel, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
            if idx % 100 == 0 or idx == len(files):
                print(f"  [{idx}/{len(files)}] パッケージング中... ({rel})", flush=True)

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    tmp_zip.rename(OUT_ZIP)

    zip_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n完了: {OUT_ZIP}", flush=True)
    print(f"ZIPサイズ: {zip_mb:.2f} MB (GitHub Release 上限 2,000 MB 以内)", flush=True)


if __name__ == "__main__":
    main()
