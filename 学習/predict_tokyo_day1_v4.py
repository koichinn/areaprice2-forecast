#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""東京エリア価格 DAY1 専用予測スクリプト v4 (v4_tokyo Super-Ensemble)
- Path A (Tokyo Direct): 東京カレンダー同期 (anc_cal) + 予報類似度 (anc_sim) + L1 (DAY/WIDE/DEEP/BLK4/BLK6) + q65/q35
- Path B (Reconstructed): System v4_sub2_super + 東京プレミアム (anc_cal_prem / anc_sim_prem / WIDE / BLOCKS6 / q65)
- Path C (Partial-Coupling Z07): 0.70 * System v4_sub2_super + Z07 (WIDE / BLOCKS6 / q65)
- 結合アンカー: 0.35 * anc_sim_tok + 0.35 * (0.7*p_sys + anc_sim_z07) + 0.30 * (anc_cal_tok + 0.7*(p_sys - anc_cal_sys))
- キャリブレーション: tokyo_day1_v4_calib.json の最適ブレンド (wq=0.70, wa=0.10)
出力: DAY1 48コマの予測値 (predicted_price_yen_kwh)
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"P:\エリアプライス2\学習")
import build_fast_eval_cache_tokyo_day1 as B
import fast_tune_tokyo_day1 as ft

INTEG = Path(r"P:\エリアプライス2\database\統合データ")
OUT_DIR = Path(r"P:\エリアプライス2\予測")
CALIB_JSON = Path(r"P:\エリアプライス2\学習\tokyo_day1_v4_calib.json")


def load_calib() -> tuple[float, float, list[str] | None, list[str] | None]:
    if CALIB_JSON.exists():
        try:
            cfg = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
            wq = float(cfg.get("wq", ft.DEFAULT_WQ))
            wa = float(cfg.get("wa", ft.DEFAULT_WA))
            tok_f = cfg.get("tok_feats")
            prem_f = cfg.get("prem_feats")
            return wq, wa, tok_f, prem_f
        except Exception:
            pass
    return ft.DEFAULT_WQ, ft.DEFAULT_WA, None, None


def predict_tokyo_day1(day: pd.Timestamp, p_sys: np.ndarray, iter_scale: float = 0.35) -> pd.DataFrame:
    idx = pd.date_range(day, periods=48, freq="30min")
    assert len(p_sys) == 48, f"p_sys must have 48 slots, got {len(p_sys)}"

    # 将来DAY1の特徴量（future_area_day1_area.parquet の東京の行）を extra として読み込む
    extra_df = None
    fa_path = INTEG / "future_area_day1_area.parquet"
    if fa_path.exists():
        try:
            fa = pd.read_parquet(fa_path)
            tok_fa = fa[(fa["area"] == "tokyo") & (pd.to_datetime(fa["target_datetime"]).isin(idx))].copy()
            if len(tok_fa) == 48:
                extra_df = tok_fa
        except Exception as e:
            print(f"WARN future_area_day1_area 読み込み: {e}")

    # 1. 過去学習データ + extra行の統合特徴量エンジニアリング
    d, aux, cand_cols, must_keep = B.load_and_engineer_tokyo(extra=extra_df)
    t = pd.to_datetime(d["target_datetime"])
    hour = t.dt.hour.to_numpy()

    # 学習期間: day - 1日以前 (リークゼロ)
    tr = (t < day.normalize() - pd.Timedelta(days=1)).to_numpy() & d["target"].notna()

    # 2. 予測対象日（day）のインデックスを取得
    te = t.isin(idx).to_numpy()
    if te.sum() != 48:
        raise RuntimeError(f"対象日 {day.date()} の48コマが特徴量データセットに見つかりません (一致コマ数: {te.sum()})")

    htr, hte = hour[tr], hour[te]
    y_tok = d["target"].to_numpy(float)
    y_sys = d["y_sys"].to_numpy(float)
    y_prem = d["y_prem"].to_numpy(float)
    y_z07 = y_tok - 0.70 * y_sys

    anc_cal_tok = aux["anc_cal_tok"]
    anc_sim_tok = aux["anc_sim_tok"]
    anc_reg_tok = aux["anc_regime_tok"]
    anc_cal_prem = aux["anc_cal_prem"]
    anc_sim_prem = aux["anc_sim_prem"]
    anc_cal_z07 = aux["anc_cal_z07"]
    anc_sim_z07 = aux["anc_sim_z07"]

    res_tok_cal = y_tok - anc_cal_tok
    res_tok_sim = y_tok - anc_sim_tok
    res_tok_lag1 = y_tok - d["area_price_yen_kwh_lag1d"].to_numpy(float)
    res_tok_lag7 = y_tok - d["area_price_yen_kwh_lag7d"].to_numpy(float)
    res_prem_cal = y_prem - anc_cal_prem
    res_prem_sim = y_prem - anc_sim_prem
    res_z07_cal = y_z07 - anc_cal_z07
    res_z07_sim = y_z07 - anc_sim_z07

    # 特徴量抽出 (calib.jsonに保存されていればキャッシュを使用)
    wq, wa, tok_feats_saved, prem_feats_saved = load_calib()
    if tok_feats_saved and all(c in d.columns for c in tok_feats_saved):
        tok_feats = tok_feats_saved
    else:
        print("  tok_feats を再計算中...")
        tok_feats = B.select_top_features(d, cand_cols, res_tok_cal, must_keep, top_k=195)

    if prem_feats_saved and all(c in d.columns for c in prem_feats_saved):
        prem_feats = prem_feats_saved
    else:
        print("  prem_feats を再計算中...")
        prem_feats = B.select_top_features(d, cand_cols, res_prem_cal, must_keep, top_k=195)

    print(f"  特徴量選択完了: tok_feats={len(tok_feats)}, prem_feats={len(prem_feats)}")
    sys.stdout.flush()

    X_tok = d[tok_feats].to_numpy(dtype=np.float32)
    X_prem = d[prem_feats].to_numpy(dtype=np.float32)

    DAY_SPLIT = [B.DAY, [h for h in range(24) if h not in B.DAY]]
    WIDE_SPLIT = [B.WIDE, [h for h in range(24) if h not in B.WIDE]]

    # 3. モデル推論
    c = {}
    t0 = time.time()
    print("  [1/3] Path A (Tokyo Direct) 推論中...")
    sys.stdout.flush()
    c["pred_tok_cal_day"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, DAY_SPLIT, iter_scale=iter_scale, seed=10)
    c["pred_tok_cal_wide"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=11)
    c["pred_tok_cal_wide_deep"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, leaf=47, lr=0.035, it=520, msl=28, seed=12)
    c["pred_tok_cal_blk4"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, B.BLOCKS4, iter_scale=iter_scale, seed=13)
    c["pred_tok_cal_blk6"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=14)
    c["pred_tok_cal_q65"] = anc_cal_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_cal[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=15)
    c["pred_tok_sim_wide"] = anc_sim_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=17)
    c["pred_tok_sim_blk6"] = anc_sim_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=18)
    c["pred_tok_sim_q65"] = anc_sim_tok[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_sim[tr], X_tok[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=19)
    c["pred_tok_lag1_wide"] = d["area_price_yen_kwh_lag1d"].to_numpy(float)[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_lag1[tr], X_tok[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=20)
    c["pred_tok_lag1_blk6"] = d["area_price_yen_kwh_lag1d"].to_numpy(float)[te] + B.blocks_fit_pred(X_tok[tr], htr, res_tok_lag1[tr], X_tok[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=21)

    print(f"  [2/3] Path B (Reconstructed Premium) 推論中... ({time.time() - t0:.1f}s)")
    sys.stdout.flush()
    c["pred_prem_cal_wide"] = anc_cal_prem[te] + B.blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=30)
    c["pred_prem_cal_blk6"] = anc_cal_prem[te] + B.blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=31)
    c["pred_prem_cal_q65"] = anc_cal_prem[te] + B.blocks_fit_pred(X_prem[tr], htr, res_prem_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=32)

    print(f"  [3/3] Path C (Partial Coupling Z07) 推論中... ({time.time() - t0:.1f}s)")
    sys.stdout.flush()
    c["pred_z07_cal_wide"] = anc_cal_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=40)
    c["pred_z07_cal_blk6"] = anc_cal_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=41)
    c["pred_z07_cal_q65"] = anc_cal_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_cal[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=42)
    c["pred_z07_sim_wide"] = anc_sim_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, WIDE_SPLIT, iter_scale=iter_scale, seed=43)
    c["pred_z07_sim_blk6"] = anc_sim_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, B.BLOCKS6, iter_scale=iter_scale, seed=44)
    c["pred_z07_sim_q65"] = anc_sim_z07[te] + B.blocks_fit_pred(X_prem[tr], htr, res_z07_sim[tr], X_prem[te], hte, WIDE_SPLIT, loss="quantile", quantile=0.65, iter_scale=iter_scale, seed=45)

    # アンカー配列
    c["anc_sim"] = anc_sim_tok[te]
    c["anc_sim_z07"] = anc_sim_z07[te]
    c["anc_cal"] = anc_cal_tok[te]
    c["anc_cal_sys"] = aux["anc_cal_sys"][te]

    # 4. スーパーアンサンブル合成
    comp = ft.components(c, p_sys)
    pred_final = ft.final_pred(comp, wq, wa)
    print(f"  全モデル推論完了 ({time.time() - t0:.1f}s)")
    sys.stdout.flush()

    res_df = pd.DataFrame({
        "target_datetime": idx,
        "area": "tokyo",
        "predicted_price_yen_kwh": pred_final,
        "direct": comp["l1"],
        "reconstructed": comp["q65"],
        "system_pred": p_sys,
        "premium_pred": pred_final - p_sys,
        "anchor_lag1d": d["area_price_yen_kwh_lag1d"].to_numpy(float)[te],
        "model_used": "v4_tokyo_super(PathA+B+C)",
        "forecast_created_at": datetime.now().astimezone().isoformat(),
        "train_rows": int(tr.sum()),
    })
    return res_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="予測対象日 (YYYY-MM-DD)")
    ap.add_argument("--system", type=Path, default=OUT_DIR / "forecast_system_day1_occto_latest.csv")
    ap.add_argument("--iter-scale", type=float, default=0.35)
    a = ap.parse_args()

    sysf = pd.read_csv(a.system, encoding="utf-8-sig")
    sysf["target_datetime"] = pd.to_datetime(sysf["target_datetime"])
    day = pd.Timestamp(a.date) if a.date else sysf["target_datetime"].min().normalize()
    idx = pd.date_range(day, periods=48, freq="30min")

    sp_series = sysf.drop_duplicates("target_datetime", keep="last").set_index("target_datetime")["predicted_system_price_yen_kwh"].reindex(idx)
    if sp_series.isna().any():
        print(f"ERROR: システム専用予測が {day.date()} の48コマ揃っていません")
        return 1

    print(f"=== 東京DAY1 v4_tokyo予測開始: DAY1={day.date()} ===", flush=True)
    t0 = time.time()
    res = predict_tokyo_day1(day, sp_series.to_numpy(float), iter_scale=a.iter_scale)
    el = time.time() - t0
    print(f"OK 東京DAY1 v4予測完了 ({el:.1f}s): 平均={res['predicted_price_yen_kwh'].mean():.2f} 最小={res['predicted_price_yen_kwh'].min():.2f} 最大={res['predicted_price_yen_kwh'].max():.2f}")

    out_p = OUT_DIR / "forecast_tokyo_day1_v4_latest.csv"
    res.to_csv(out_p, index=False, encoding="utf-8-sig")
    print(f"保存完了: {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
