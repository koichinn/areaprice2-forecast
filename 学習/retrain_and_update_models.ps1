#requires -Version 5.1
<#
.SYNOPSIS
  ローカルPC専用: マスターDB全項目（9社需給・JEPX・OCCTO・気象・統合データ）の完全更新 ＆ AIモデル再学習スクリプト
.DESCRIPTION
  定期的（月1回や季節の変わり目など）にローカルPCで実行することで、
  1. 送配電9社の需給実績マスター
  2. JEPX約定価格・入札量マスター
  3. OCCTO（広域機関）予備率・連系線マスター
  4. 天気9エリアの気象実績・予報データ
  5. 過去統合データ (past_integrated_9area_30min.parquet)
  6. 将来統合データ (future_integrated_9area_30min.parquet)
  7. 学習チェーン (training_v8_price_regime/*.parquet)
  8. AI軽量推論モデル群 (models/*.joblib)
  の全項目を漏れなく1本で完全に最新化します。
#>

[CmdletBinding()]
param(
  [int]$SupplyLookbackDays = 45,
  [switch]$SkipModelRetrain
)

$ErrorActionPreference = "Stop"

# P: ドライブ確認（未マウント時は自動マウント）
$targetDir = "C:\Users\中西孝一\OneDrive - NTTアノードエナジー株式会社\ドキュメント\program\エリアプライス2"
if (!(Test-Path "P:\")) {
  Write-Host "[Info] P: ドライブを自動マウントします: $targetDir" -ForegroundColor Cyan
  subst P: $targetDir
}

$Root = "P:\エリアプライス2"
$Database = Join-Path $Root "database"
$Learning = Join-Path $Root "学習"
$Occto = Join-Path $Database "occtoデータ"
$Weather = Join-Path $Database "天気9エリア"
$Supply = Join-Path $Database "送配電９社データ"

function Run-Step([string]$title, [string]$scriptPath, [string[]]$pyArgs = @()) {
  Write-Host "`n==========================================================" -ForegroundColor Cyan
  Write-Host " $title" -ForegroundColor Cyan
  Write-Host "==========================================================" -ForegroundColor Cyan
  if (!(Test-Path $scriptPath)) {
    throw "スクリプトが見つかりません: $scriptPath"
  }
  $old = Get-Location
  try {
    Set-Location (Split-Path -Parent $scriptPath)
    & python (Split-Path -Leaf $scriptPath) @pyArgs
    if ($LASTEXITCODE -ne 0) {
      throw "$title がエラーコード $LASTEXITCODE で終了しました。"
    }
  } finally {
    Set-Location $old
  }
}

$t0 = Get-Date

Write-Host "`n**********************************************************" -ForegroundColor Green
Write-Host " [マスターDB全項目 一括更新 & AIモデル再学習] 開始" -ForegroundColor Green
Write-Host " 実行日時: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
Write-Host "**********************************************************" -ForegroundColor Green

# 1. 送配電9社 需給実績マスター
Run-Step "[1/8] 送配電9社 需給実績マスター更新 (area_supply_master_v2.parquet)" `
  (Join-Path $Supply "update_area_supply_master.py") `
  @("--lookback-days", "$SupplyLookbackDays", "--daily-stale-days", "30", "--hokkaido-stale-days", "30", "--shikoku-stale-days", "30", "--chubu-monthly-grace-days", "31")

# 2. JEPX約定価格・入札量マスター
Run-Step "[2/8-A] JEPX 年別CSVダウンロード" (Join-Path $Learning "download_jepx_spot_summary.py")
Run-Step "[2/8-B] JEPX 実績マスター更新 (jepx_spot_summary_master.csv)" (Join-Path $Learning "update_jepx_spot_master.py")

# 3. OCCTO 広域予備率・連系線データ
$occtoDownload = Join-Path $Occto "download_occto_dayaftertomorrow_3datasets_requests_final.py"
if (Test-Path $occtoDownload) {
  try { Run-Step "[3/8-A] OCCTO 翌々日3データ取得" $occtoDownload } catch { Write-Host "WARN OCCTO取得スキップ（既存データで続行）: $($_.Exception.Message)" -ForegroundColor Yellow }
}
$occtoInter = Join-Path $Occto "download_occto_interconnection_tomorrow_dayaftertomorrow.py"
if (Test-Path $occtoInter) {
  try { Run-Step "[3/8-B] OCCTO 連系線空容量取得" $occtoInter } catch { Write-Host "WARN 連系線取得スキップ: $($_.Exception.Message)" -ForegroundColor Yellow }
}
Run-Step "[3/8-C] OCCTO 特徴量ファイル更新 (occto_da8.parquet / occto_area_ic_da.parquet)" (Join-Path $Learning "update_occto_da8.py")

# 4. 天気9エリア 気象データ
$weatherScript = Join-Path $Weather "update_weather_forecast_9area_30min.py"
if (Test-Path $weatherScript) {
  Run-Step "[4/8] 天気9エリア 気象予報・実績更新" $weatherScript
}

# 5. 過去統合データの再生成 (需給 + JEPX + OCCTO + 気象の全結合)
Run-Step "[5/8] 過去統合データ再生成 (past_integrated_9area_30min.parquet)" (Join-Path $Learning "integrate_past_data.py")

# 6. 将来統合データの再生成
Run-Step "[6/8] 将来統合データ再生成 (future_integrated_9area_30min.parquet)" (Join-Path $Learning "integrate_future_data_occto_v3.py")

# 7. 全9エリア 学習チェーンの再構築
Run-Step "[7/8] 学習チェーン再構築 (training_v8_price_regime/*.parquet)" (Join-Path $Learning "rebuild_training_chain_v8.py")

# 8. AI軽量推論モデル群の再学習＆保存
if (!$SkipModelRetrain) {
  Run-Step "[8/8] 全9エリア AI推論モデル再学習＆軽量エクスポート (models/*.joblib)" (Join-Path $Learning "export_cloud_inference_models.py")
} else {
  Write-Host "`n[8/8] モデル再学習はスキップされました (-SkipModelRetrain)" -ForegroundColor Yellow
}

$elapsed = (Get-Date) - $t0
Write-Host "`n**********************************************************" -ForegroundColor Green
Write-Host " [完了] マスターDB全項目の更新が正常に完了しました！" -ForegroundColor Green
Write-Host " 総所要時間: $($elapsed.TotalMinutes.ToString('N1')) 分" -ForegroundColor Green
Write-Host " models/ を更新した場合は、Gitリポジトリへコミット＆プッシュしてください。" -ForegroundColor Green
Write-Host "**********************************************************" -ForegroundColor Green
