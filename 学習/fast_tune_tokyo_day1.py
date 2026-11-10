#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""東京エリア価格 DAY1 高速チューニング＆検証 (v4_tokyo)
oof_cache_tokyo_day1.parquet (+ 任意で oof_cache_tokyo_split.parquet) を読み込み、
L1/q65 3系統スーパーアンサンブル + 結合アンカーの最終予測を作り、6 foldウォークフォワード検証する。
ブレンド比 (wq, wa) は「過去foldのみ」で選ぶ walk-forward 版も同時に評価 (リークゼロ確認用)。
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path(r"P:\エリアプライス2\学習\fast_eval_cache")
CACHE_FILE = CACHE_DIR / "oof_cache_tokyo_day1.parquet"
SPLIT_FILE = CACHE_DIR / "oof_cache_tokyo_split.parquet"
CALIB_JSON = Path(r"P:\エリアプライス2\学習\tokyo_day1_v4_calib.json")

# 本番と共有するブレンド定義 (predict_tokyo_day1_v4.py からも import)
L1_W = {
    "tok_cal": 0.30, "tok_sim": 0.15, "tok_lag1": 0.10,
    "rec_cal": 0.15, "z07_cal": 0.15, "z07_sim": 0.15,
}
Q65_W = {"tok_cal_q65": 0.30, "tok_sim_q65": 0.20, "rec_cal_q65": 0.15, "z07_cal_q65": 0.20, "z07_sim_q65": 0.15}
DEFAULT_WQ = 0.70
DEFAULT_WA = 0.10


def components(c: dict, p_sys: np.ndarray) -> dict:
    """c: pred_* 配列の dict (キャッシュ/本番共通の列名)"""
    tok_cal = 0.20 * c["pred_tok_cal_day"] + 0.25 * c["pred_tok_cal_wide"] + 0.25 * c["pred_tok_cal_wide_deep"] + 0.15 * c["pred_tok_cal_blk4"] + 0.15 * c["pred_tok_cal_blk6"]
    tok_sim = 0.55 * c["pred_tok_sim_wide"] + 0.45 * c["pred_tok_sim_blk6"]
    tok_lag1 = 0.60 * c["pred_tok_lag1_wide"] + 0.40 * c["pred_tok_lag1_blk6"]
    rec_cal = p_sys + 0.55 * c["pred_prem_cal_wide"] + 0.45 * c["pred_prem_cal_blk6"]
    z07_cal = 0.70 * p_sys + 0.55 * c["pred_z07_cal_wide"] + 0.45 * c["pred_z07_cal_blk6"]
    z07_sim = 0.70 * p_sys + 0.55 * c["pred_z07_sim_wide"] + 0.45 * c["pred_z07_sim_blk6"]
    l1 = (L1_W["tok_cal"] * tok_cal + L1_W["tok_sim"] * tok_sim + L1_W["tok_lag1"] * tok_lag1
          + L1_W["rec_cal"] * rec_cal + L1_W["z07_cal"] * z07_cal + L1_W["z07_sim"] * z07_sim)
    q65 = (Q65_W["tok_cal_q65"] * c["pred_tok_cal_q65"] + Q65_W["tok_sim_q65"] * c["pred_tok_sim_q65"]
           + Q65_W["rec_cal_q65"] * (p_sys + c["pred_prem_cal_q65"])
           + Q65_W["z07_cal_q65"] * (0.70 * p_sys + c["pred_z07_cal_q65"])
           + Q65_W["z07_sim_q65"] * (0.70 * p_sys + c["pred_z07_sim_q65"]))
    anc = (0.35 * c["anc_sim"] + 0.35 * (0.70 * p_sys + c["anc_sim_z07"])
           + 0.30 * (c["anc_cal"] + 0.70 * (p_sys - c["anc_cal_sys"])))
    return {"l1": l1, "q65": q65, "anc": anc}


def final_pred(comp: dict, wq: float, wa: float) -> np.ndarray:
    return (1.0 - wa) * ((1.0 - wq) * comp["l1"] + wq * comp["q65"]) + wa * comp["anc"]


def wmae(y, p, folds):
    fm = np.array([np.mean(np.abs(y[folds == k] - p[folds == k])) for k in sorted(np.unique(folds))])
    n = np.array([np.sum(folds == k) for k in sorted(np.unique(folds))])
    return float(np.average(fm, weights=n)), fm


def main():
    t0 = time.time()
    df = pd.read_parquet(CACHE_FILE).sort_values("target_datetime").reset_index(drop=True)
    y = df["y"].to_numpy(float)
    folds = df["fold_idx"].to_numpy()
    p_sys = df["p_sys_v4"].to_numpy(float)
    c = {k: df[k].to_numpy(float) for k in df.columns if k.startswith(("pred_", "anc_"))}
    comp = components(c, p_sys)

    fixed = final_pred(comp, DEFAULT_WQ, DEFAULT_WA)

    WQ = [0.5, 0.6, 0.7, 0.8]; WA = [0.0, 0.1, 0.2]
    wf = fixed.copy(); chosen = {}
    for k in sorted(np.unique(folds)):
        cur = folds == k
        if k == 0:
            chosen[int(k)] = (DEFAULT_WQ, DEFAULT_WA); continue
        past = folds < k
        best = min(((wq, wa) for wq in WQ for wa in WA), key=lambda z: np.mean(np.abs(y[past] - final_pred({kk: v[past] for kk, v in comp.items()}, *z))))
        chosen[int(k)] = best
        wf[cur] = final_pred({kk: v[cur] for kk, v in comp.items()}, *best)

    rows = {"lag1": df["lag1"].to_numpy(float), "anc_cal": df["anc_cal"].to_numpy(float),
            "l1_super": comp["l1"], "q65_super": comp["q65"], "anc_coupled": comp["anc"],
            "v4_tokyo_fixed": fixed, "v4_tokyo_wf": wf}

    if SPLIT_FILE.exists():
        sp = pd.read_parquet(SPLIT_FILE)
        sp["target_datetime"] = pd.to_datetime(sp["target_datetime"])
        sp = sp.set_index("target_datetime").reindex(pd.to_datetime(df["target_datetime"]))
        cs = 0.55 * sp["pred_cs_wide"].to_numpy(float) + 0.45 * sp["pred_cs_blk6"].to_numpy(float)
        fc = 0.55 * sp["pred_fc_wide"].to_numpy(float) + 0.45 * sp["pred_fc_blk6"].to_numpy(float)
        split_l1 = p_sys + cs + fc
        split_q = p_sys + sp["pred_cs_q65"].to_numpy(float) + sp["pred_fc_q65"].to_numpy(float)
        rows["split_l1"] = split_l1
        rows["split_q65"] = split_q
        for ws in (0.15, 0.25, 0.35):
            rows[f"fixed+split{int(ws*100)}"] = (1 - ws) * fixed + ws * (0.4 * split_l1 + 0.6 * split_q)

    res = []
    for nm, p in rows.items():
        m, fm = wmae(y, p, folds)
        res.append((m, nm, fm))
    print(f"=== 東京DAY1 高速チューニング ({time.time()-t0:.2f}s) ===")
    for m, nm, fm in sorted(res):
        print(f"  {nm:22s} W_MAE={m:.4f}  folds={np.round(fm, 3).tolist()}")
    print("walk-forward 選択 (wq, wa):", chosen)
    CALIB_JSON.write_text(json.dumps({"wq": DEFAULT_WQ, "wa": DEFAULT_WA, "L1_W": L1_W, "Q65_W": Q65_W,
                                      "wf_weighted_mae": wmae(y, wf, folds)[0], "fixed_weighted_mae": wmae(y, fixed, folds)[0]},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
