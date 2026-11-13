# -*- coding: utf-8 -*-

"""

予測精度ダッシュボードHTMLを生成する。

対象日より前に作成された予測だけを候補とし、

target_date / slot / area ごとに forecast_created_at が最新の予測を採用する。

実績確定後の事後予測は除外する。

"""

from pathlib import Path

import json

import pandas as pd



WEB = Path(r"P:\エリアプライス2\web")

ACCURACY_CSV = WEB / "forecast_day1_to_day7_accuracy.csv"

TEMPLATE = WEB / "accuracy_dashboard_web_template.html"

OUTPUT = WEB / "forecast_accuracy_dashboard.html"



KEYS = ["target_date", "slot", "area"]





def read_csv_auto(path):

    last = None

    for enc in ("utf-8-sig", "cp932", "utf-8"):

        try:

            return pd.read_csv(path, encoding=enc, low_memory=False)

        except UnicodeDecodeError as exc:

            last = exc

    raise RuntimeError(f"文字コードを判定できません: {path}") from last





def main():

    for path in (ACCURACY_CSV, TEMPLATE):

        if not path.is_file():

            raise FileNotFoundError(path)



    raw = read_csv_auto(ACCURACY_CSV)



    required = {

        "target_date",

        "slot",

        "area",

        "forecast_created_at",

        "predicted_price_yen_kwh",

        "actual_price_yen_kwh",

    }



    missing = required - set(raw.columns)

    if missing:

        raise KeyError(f"必須列不足: {sorted(missing)}")



    raw["target_date"] = (

        pd.to_datetime(

            raw["target_date"],

            errors="coerce"

        ).dt.normalize()

    )



    raw["_created_at"] = pd.to_datetime(raw["forecast_created_at"], errors="coerce", utc=True, format="mixed")



    # forecast_created_at を JST 日付へ変換

    raw["_created_date"] = (

        raw["_created_at"]

        .dt.tz_convert("Asia/Tokyo")

        .dt.tz_localize(None)

        .dt.normalize()

    )



    has_prediction = raw["predicted_price_yen_kwh"].notna()



    # forecast_created_at の日付 < target_date

    eligible = raw[

        has_prediction

        & raw["_created_date"].notna()

        & (raw["_created_date"] < raw["target_date"])

    ].copy()



    selected = (

        eligible

        .sort_values(

            KEYS + ["_created_at"]

        )

        .drop_duplicates(

            KEYS,

            keep="last"

        )

    )



    # 画面表示用の採用計算日

    selected["selected_calc_date"] = (

        selected["_created_at"]

        .dt.tz_convert("Asia/Tokyo")

        .dt.strftime("%Y-%m-%d")

    )



    actual = (

        raw[

            raw["actual_price_yen_kwh"].notna()

        ]

        .sort_values(

            KEYS + ["_created_at"],

            na_position="first"

        )

        .drop_duplicates(

            KEYS,

            keep="last"

        )

        [KEYS + ["actual_price_yen_kwh"]]

    )



    keep = [

        c

        for c in raw.columns

        if not c.startswith("_")

        and c != "actual_price_yen_kwh"

    ]



    # HTMLへ渡す表示用列

    keep.append("selected_calc_date")

    # --- 計算ラン(1回の実行)の識別: DAY1日付ごとに、計算時刻が15分以内のまとまりを1ランとする ---

    _d = pd.to_numeric(raw["day"], errors="coerce") if "day" in raw.columns else pd.Series(float("nan"), index=raw.index)

    raw["_day1"] = raw["target_date"] - pd.to_timedelta((_d - 1).fillna(0), unit="D")

    raw["_cj"] = raw["_created_at"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

    _runs = raw.loc[raw["_cj"].notna() & _d.notna(), ["_day1", "_cj"]].drop_duplicates().sort_values(["_day1", "_cj"])

    _runs["_new"] = (_runs["_day1"] != _runs["_day1"].shift()) | (_runs["_cj"].diff() > pd.Timedelta(minutes=15))

    _runs["_run_start"] = _runs.groupby(_runs["_new"].cumsum())["_cj"].transform("min")

    selected["_day1"] = selected["target_date"] - pd.to_timedelta((pd.to_numeric(selected["day"], errors="coerce") - 1).fillna(0), unit="D") if "day" in selected.columns else selected["target_date"]

    selected["_cj"] = selected["_created_at"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

    selected = selected.merge(_runs[["_day1", "_cj", "_run_start"]], on=["_day1", "_cj"], how="left")

    selected["selected_run"] = selected["_run_start"].dt.strftime("%Y-%m-%d %H:%M").fillna(selected["selected_calc_date"])

    selected["selected_day1"] = selected["_day1"].dt.strftime("%Y-%m-%d")

    keep.append("selected_run")

    keep.append("selected_day1")



    result = (

        selected[keep]

        .merge(

            actual,

            on=KEYS,

            how="outer"

        )

    )



    result["target_date"] = (

        pd.to_datetime(

            result["target_date"],

            errors="coerce"

        )

        .dt.strftime("%Y-%m-%d")

    )



    result["error"] = (

        result["predicted_price_yen_kwh"]

        - result["actual_price_yen_kwh"]

    )



    result["abs_error"] = result["error"].abs()



    market = actual["target_date"].max()



    market_str = (

        pd.to_datetime(market).strftime("%Y-%m-%d")

        if pd.notna(market)

        else None

    )



    payload = {

        "marketConfirmedThrough": market_str,

        "selectionRule":

            "forecast_created_at の日付 < target_date のうち最新",

        "records": json.loads(

            result.to_json(

                orient="records",

                date_format="iso"

            )

        ),

    }



    template = TEMPLATE.read_text(

        encoding="utf-8"

    )



    if "__PAYLOAD__" not in template:

        raise RuntimeError(

            f"テンプレートに __PAYLOAD__ がありません: "

            f"{TEMPLATE}"

        )



    payload_json = json.dumps(

        payload,

        ensure_ascii=False,

        separators=(",", ":")

    )



    OUTPUT.write_text(

        template.replace(

            "__PAYLOAD__",

            payload_json

        ),

        encoding="utf-8"

    )



    excluded = int(

        has_prediction.sum()

        - len(eligible)

    )



    print(f"SUCCESS output={OUTPUT}")

    print(

        f"records={len(result):,} "

        f"predictions_selected={len(selected):,} "

        f"post_target_excluded={excluded:,}"

    )

    print(

        f"marketConfirmedThrough={market_str}"

    )





if __name__ == "__main__":

    main()



