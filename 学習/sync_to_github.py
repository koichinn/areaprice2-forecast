#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub 本番リポジトリ (koichinn/areaprice2-forecast) へ予測プログラムの更改ファイルを直接プッシュするスクリプト。
Git (git.exe) 不要で Python 標準ライブラリ + requests のみで GitHub REST API 経由で main ブランチを更新します。

トークンの優先順位:
1. 環境変数 GITHUB_TOKEN または GH_TOKEN
2. P:\\エリアプライス2\\.github_token.txt (1行目に Fine-grained PAT または Classic PAT を記載、Git・ZIPパッケージ対象外)
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path
import requests
import truststore

truststore.inject_into_ssl()

ROOT = Path(r"P:\エリアプライス2")
TOKEN_FILE = ROOT / ".github_token.txt"
REPO = "koichinn/areaprice2-forecast"
BRANCH = "main"

DEFAULT_SYNC_FILES = [
    ".github/workflows/daily_forecast.yml",
    "学習/predict_system_day1_occto.py",
    "学習/system_day1_v3_calib.json",
    "学習/fast_tune_system_day1.py",
    "学習/build_future_weather_features_system.py",
    "学習/build_future_area_features.py",
    "学習/predict_area_day1_recon.py",
    "学習/predict_tokyo_day1_v4.py",
    "学習/predict_tokyo_day1_v4_fast.py",
    "学習/build_fast_eval_cache_tokyo_day1.py",
    "学習/fast_tune_tokyo_day1.py",
    "学習/tokyo_day1_v4_calib.json",
    "学習/export_cloud_inference_models.py",
    "学習/integrate_past_data.py",
    "学習/occto_area_ic_updater.py",
    "学習/update_occto_da8.py",
    "学習/replace_area_day1.py",
    "学習/run_daily_forecast_occto_v4.ps1",
    "学習/retrain_and_update_models.ps1",
    "学習/build_cloud_package.py",
    "仕様/クラウド自動実行_設定手順書.md",
]


def load_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not tok and TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
    return tok


def push_file(session: requests.Session, rel_path: str, commit_msg: str) -> bool:
    local_p = ROOT / rel_path
    if not local_p.is_file():
        print(f"[SKIP] ファイルが存在しません: {local_p}")
        return False

    api_url = f"https://api.github.com/repos/{REPO}/contents/{rel_path}"
    r_get = session.get(api_url, params={"ref": BRANCH}, timeout=30)
    sha = None
    if r_get.status_code == 200:
        obj = r_get.json()
        sha = obj.get("sha")
        remote_b64 = (obj.get("content") or "").replace("\n", "")
        local_bytes = local_p.read_bytes()
        if base64.b64encode(local_bytes).decode("ascii") == remote_b64:
            print(f"[SAME] 変更なし: {rel_path}")
            return True
    elif r_get.status_code != 404:
        print(f"[ERR ] GET {rel_path} failed ({r_get.status_code}): {r_get.text[:200]}")
        return False

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(local_p.read_bytes()).decode("ascii"),
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    r_put = session.put(api_url, json=payload, timeout=60)
    if r_put.status_code in (200, 201):
        print(f"[OK  ] GitHub 本番反映完了: {rel_path}")
        return True
    print(f"[ERR ] PUT {rel_path} failed ({r_put.status_code}): {r_put.text[:200]}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="同期する相対パス（省略時は標準の予測プログラム一式）")
    ap.add_argument("-m", "--message", default="Update forecast program to v4_sub2_super (MAE 1.9991)", help="Commit message")
    a = ap.parse_args()

    tok = load_token()
    if not tok:
        print(f"ERROR: GitHub トークンが未設定です。{TOKEN_FILE} に Personal Access Token を1行保存してください。")
        return 1

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    targets = a.files if a.files else DEFAULT_SYNC_FILES
    ok_all = True
    for rel in targets:
        rel_norm = rel.replace("\\", "/")
        if not push_file(session, rel_norm, a.message):
            ok_all = False
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
