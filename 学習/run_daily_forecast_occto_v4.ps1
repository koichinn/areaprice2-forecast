#requires -Version 5.1

[CmdletBinding()]

param([switch]$SkipFreshnessCheck,[switch]$SkipSupplyUpdate,[int]$OcctoRetryCount=3,[int]$OcctoRetrySeconds=45,[int]$SupplyStaleLimitDays=30,[int]$PastStopDays=7,[switch]$ForceTrainingChain,[switch]$SkipSystemModel,[switch]$SkipSystemReplace,[switch]$SkipAreaModel,[switch]$ApplyAreaReplace,[switch]$SkipAreaReplace)

$ErrorActionPreference="Stop"

$script:Warns=@()

$Learning="P:\エリアプライス2\学習"; $Database="P:\エリアプライス2\database"

$Occto=Join-Path $Database "occtoデータ"; $Downloads=Join-Path $Occto "downloads"

$SupplyDir=(Get-ChildItem $Database -Directory | Where-Object Name -like "送配電*" | Select-Object -First 1).FullName

$LogDir=Join-Path $Learning "logs"; New-Item -ItemType Directory -Force $LogDir|Out-Null

$Log=Join-Path $LogDir ("run_daily_forecast_occto_v4_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Log([string]$s,[string]$c="Gray"){Add-Content $Log $s -Encoding UTF8;Write-Host $s -ForegroundColor $c;if($c -eq 'Yellow' -or $c -eq 'Red'){$script:Warns+=$s.Trim()}}

function Run-Python([string]$Name,[string]$Script,$PyArgs=@()){if(!(Test-Path $Script)){throw "$Name script not found: $Script"};Log "`n=== $Name ===" Cyan;$old=Get-Location;Set-Location (Split-Path -Parent $Script);$save=$ErrorActionPreference;$ErrorActionPreference='Continue';try{& python $Script @PyArgs 2>&1|%{$x="$_";Add-Content $Log $x -Encoding UTF8;Write-Host $x};$code=$LASTEXITCODE}finally{Set-Location $old;$ErrorActionPreference=$save};if($code-ne 0){throw "$Name failed exit=$code"}}

function Find-Cache([string[]]$Patterns,[datetime]$Target){$stamp=$Target.ToString('yyyyMMdd');foreach($pat in $Patterns){$f=Get-ChildItem $Downloads -File -ErrorAction SilentlyContinue|?{$_.Name-like $pat-and$_.Name-match $stamp}|sort LastWriteTime -Descending|select -First 1;if($f){return $f}};$null}

function Get-OcctoRule {

 $now=Get-Date;$before18=$now.Hour-lt 18

 # OCCTO取得プログラムと同じ規則: 18時前の「翌々日」対象は暦日+1、18時以降は暦日+2

 $targetDay2=if($before18){$now.Date.AddDays(1)}else{$now.Date.AddDays(2)}

 [pscustomobject]@{Before18=$before18;Tomorrow=$now.Date.AddDays(1);TargetDay2=$targetDay2}

}

function Assert-DayafterCache {

 $r=Get-OcctoRule;$required=@(@{Label='広域ブロック情報(翌々日)';Patterns=@('*occto_wide_block_reserve_dayaftertomorrow*.csv')})

 if(!$r.Before18){$required+=@(@{Label='エリア・広域ブロック情報(翌々日)';Patterns=@('*occto_area_wide_block_reserve_dayaftertomorrow*.csv')},@{Label='電力需要予想ピーク時供給力(翌々日)';Patterns=@('*occto_demand_peak_supply_dayaftertomorrow*.csv')})}

 else{Log "OCCTO 18:00ルール: 対象日=$($r.TargetDay2.ToString('yyyy-MM-dd'))。エリア・広域ブロック情報とピーク時供給力は任意" Yellow}

 $missing=@();$used=@();foreach($x in $required){$f=Find-Cache $x.Patterns $r.TargetDay2;if($f){$used+="$($x.Label)=$($f.Name)"}else{$missing+="$($x.Label) target=$($r.TargetDay2.ToString('yyyy-MM-dd'))"}}

 if($missing.Count){throw "OCCTO翌々日必須キャッシュ不足: $($missing-join '; ')"};Log ("OCCTO翌々日 cache accepted: "+($used-join ' | ')) Yellow

}

function Assert-InterconnectionCache {

 $r=Get-OcctoRule;$required=@(

  @{Label='連系線空容量(翌日)';Target=$r.Tomorrow;Patterns=@('*interconnection*tomorrow*.csv','*interconnection*nextday*.csv')},

  @{Label='連系線空容量(翌々日)';Target=(Get-Date).Date.AddDays(2);Patterns=@('*interconnection*dayaftertomorrow*.csv')})

 $missing=@();$used=@();foreach($x in $required){$f=Find-Cache $x.Patterns $x.Target;if($f){$used+="$($x.Label)=$($f.Name)"}else{$missing+="$($x.Label) target=$($x.Target.ToString('yyyy-MM-dd'))"}}

 if($missing.Count){throw "OCCTO連系線必須キャッシュ不足: $($missing-join '; ')"};Log ("OCCTO連系線 cache accepted: "+($used-join ' | ')) Yellow

}

function Invoke-NativePython([string]$Script,$PyArgs=@()){

 $save=$ErrorActionPreference;$oldLocation=Get-Location;$ErrorActionPreference='Continue'

 try{

  Set-Location (Split-Path -Parent $Script)

  $o=@(& python $Script @PyArgs 2>&1);$code=$LASTEXITCODE

 } finally {

  Set-Location $oldLocation

  $ErrorActionPreference=$save

 }

 $o|%{Add-Content $Log $_ -Encoding UTF8;Write-Host $_}

 [pscustomobject]@{Code=$code;Text=($o-join"`n")}

}

# [V4.7追加] 9エリア需給マスター更新

function Update-SupplyMaster {

 if(!$SupplyDir){throw "送配電９社データ フォルダが見つかりません: $Database"}

 $script=Join-Path $SupplyDir 'update_area_supply_master.py'

 if(!(Test-Path $script)){throw "0 需給マスター更新 script not found: $script"}

 Log "`n=== 0 需給マスター更新 ===" Cyan

 $ha=@('--daily-stale-days',"$SupplyStaleLimitDays",'--hokkaido-stale-days',"$SupplyStaleLimitDays",'--shikoku-stale-days',"$SupplyStaleLimitDays",'--chubu-monthly-grace-days','31')

 $res=Invoke-NativePython $script $ha 

 if($res.Code-ne 0){throw "0 需給マスター更新 failed exit=$($res.Code)"}

}

# [V4.7追加] エリア別鮮度チェック（1エリアでもNGなら停止）

function Assert-SupplyFreshness {

 $rp=Join-Path $SupplyDir 'update_area_supply_master_report.json'

 if(!(Test-Path $rp)){throw "需給レポートがありません: $rp"}

 $age=(Get-Date)-(Get-Item $rp).LastWriteTime

 if($age.TotalHours-gt 6){throw ("需給レポートが古い（{0:N1}時間前）。update_area_supply_master.py が実行されていません" -f $age.TotalHours)}

 $r=Get-Content $rp -Raw -Encoding UTF8|ConvertFrom-Json

 if($r.status-ne 'SUCCESS'){throw "需給マスター status=$($r.status) error=$($r.error)"}

 $ng=@()

 foreach($p in $r.freshness.PSObject.Properties){

  $v=$p.Value

  $line="{0}: mode={1} latest={2} status={3}" -f $p.Name,$v.mode,$v.latest,$v.status

  if($v.status-ne 'OK'){$ng+=$line;Log "NG $line" Red}else{Log "OK $line" Green}

  if($v.age_days -gt 3){Log ("WARN {0}: 最終日から{1}日経過（既定の上限4日を超過）" -f $p.Name,$v.age_days) Yellow}

 }

 if($r.gaps){foreach($p in $r.gaps.PSObject.Properties){if([int]$p.Value.missing_slots-gt 0){Log ("WARN {0}: 直近45日の欠損={1}スロット" -f $p.Name,$p.Value.missing_slots) Yellow}}}

 if($ng.Count){throw "需給データ鮮度NGのエリアあり: $($ng-join '; ')"}

}

try{

 if(!$SkipSupplyUpdate){try{Update-SupplyMaster}catch{Log "WARN 需給マスター更新に失敗（既存マスターで続行）: $($_.Exception.Message)" Yellow}}else{Log "0 需給マスター更新 SKIP (-SkipSupplyUpdate)" Yellow}

 if(!$SkipFreshnessCheck){try{Assert-SupplyFreshness}catch{Log "WARN 需給鮮度チェック: $($_.Exception.Message)" Yellow}}

 try{Run-Python '1a JEPX年別CSV取得' (Join-Path $Learning 'download_jepx_spot_summary.py')}catch{Log "WARN JEPX取得失敗（既存CSVで続行）: $($_.Exception.Message)" Yellow}

 Run-Python '1 JEPX実績master更新' (Join-Path $Learning 'update_jepx_spot_master.py')

 $jm=& python -c "import pandas as pd;d=pd.read_csv(r'P:\エリアプライス2\database\JPEXデータ\data\jepx_spot_summary_master.csv',encoding='utf-8-sig');c=[x for x in d.columns if '受渡日' in x][0];print(pd.to_datetime(d[c]).max().strftime('%Y-%m-%d'))"

 $JepxMax=Get-Date (($jm|select -Last 1).Trim())

 $JepxNeed=if((Get-Date).Hour -lt 11){(Get-Date).Date}else{(Get-Date).Date.AddDays(1)}

 if($JepxMax -lt $JepxNeed){Log ("WARN JEPX master max={0:yyyy-MM-dd} < 期待={1:yyyy-MM-dd}（11:00ルール）" -f $JepxMax,$JepxNeed) Yellow}else{Log ("JEPX master max={0:yyyy-MM-dd} OK" -f $JepxMax) Green}

 Run-Python '2 過去統合データ再生成' (Join-Path $Learning 'integrate_past_data.py')

 $Past=Join-Path $Database '統合データ\past_integrated_9area_30min.parquet'

 $m=& python -c "import pandas as pd;d=pd.read_parquet(r'$Past',columns=['datetime']);print(pd.to_datetime(d['datetime']).max().strftime('%Y-%m-%d %H:%M:%S'))"

 $MaxDate=Get-Date (($m|select -Last 1).Trim())

 Log "past_integrated max=$($MaxDate.ToString('yyyy-MM-dd HH:mm:ss'))" Green

 $pastAge=((Get-Date).Date-$MaxDate.Date).Days

 if($pastAge -gt $PastStopDays -and !$SkipFreshnessCheck){throw "過去統合データが古いため予測中止（最終日$($MaxDate.ToString('yyyy-MM-dd'))、$pastAge日前）"}elseif($pastAge -gt 1){Log "WARN 過去統合データの最終日が$pastAge日前（直近の行が不足）" Yellow}

 try{

 $download=Join-Path $Occto 'download_occto_dayaftertomorrow_3datasets_requests_final.py';$ok=$false;$last=''

 for($i=1;$i-le$OcctoRetryCount;$i++){

  Log "`n=== 3 OCCTO翌々日取得 attempt $i/$OcctoRetryCount ===" Cyan;$res=Invoke-NativePython $download

  if($res.Code-eq 0){$ok=$true;break};$last=$res.Text

  $rule=Get-OcctoRule

  if($rule.Before18-and$last-match 'print_32|\[print\] 32|HTMLが返却|タイムアウト'){

   try{Assert-DayafterCache;Log '18:00前の公開前応答として即時フォールバックします。' Yellow;$ok=$true;break}catch{Log $_.Exception.Message Yellow}

  }

  if($i-lt$OcctoRetryCount){Start-Sleep $OcctoRetrySeconds}

 }

 if(!$ok){if($last-notmatch '503|Service Unavailable|429|timeout|timed out|タイムアウト|再度、ログイン|JSON以外|HTMLが返却|CF01S010Pなし'){throw "OCCTO取得が非一時エラー: $last"};Assert-DayafterCache}

 $inter=Join-Path $Occto 'download_occto_interconnection_tomorrow_dayaftertomorrow.py'

 if(Test-Path $inter){$ir=Invoke-NativePython $inter;if($ir.Code-ne 0){Log '連系線取得失敗。18:00ルールでキャッシュ検証へ切替。' Yellow;Assert-InterconnectionCache}}

}catch{Log "WARN OCCTO取得に失敗（予測は続行）: $($_.Exception.Message)" Yellow}

 Run-Python '4 未来気象更新' (Join-Path $Database '天気9エリア\update_weather_forecast_9area_30min.py')

 Run-Python '5 future統合' (Join-Path $Learning 'integrate_future_data_occto_v3.py')

 $Fut=Join-Path $Database '統合データ\future_integrated_9area_30min.parquet'

 $fm=& python -c "import pandas as pd;d=pd.read_parquet(r'$Fut',columns=['datetime']);s=pd.to_datetime(d['datetime']);print(s.min().strftime('%Y-%m-%d %H:%M'));print(s.max().strftime('%Y-%m-%d %H:%M'))"

 $FutStart=Get-Date (($fm|select -Last 2|select -First 1).Trim()); $FutEnd=Get-Date (($fm|select -Last 1).Trim())

 $now=Get-Date

 # JEPX 11:00ルール: 11時前=翌日がDAY1、11時以降=翌々日がDAY1

 $Day1=if($now.Hour -lt 11){$now.Date.AddDays(1)}else{$now.Date.AddDays(2)}

 $Day7End=$Day1.AddDays(7).AddMinutes(-30)

 Log ("DAY1={0} DAY7={1} (実行 {2:HH:mm}, 11:00ルール) future={3:yyyy-MM-dd HH:mm}..{4:yyyy-MM-dd HH:mm}" -f $Day1.ToString('yyyy-MM-dd'),$Day1.AddDays(6).ToString('yyyy-MM-dd'),$now,$FutStart,$FutEnd) Green

 if($FutStart -gt $Day1 -or $FutEnd -lt $Day7End){throw "futureがDAY1-DAY7を網羅していない。気象/future統合が古いため予測中止"}

 if($now.Hour -ge 11 -and $now.Hour -lt 18){Log "WARN 11:00-18:00実行: DAY1($($Day1.ToString('yyyy-MM-dd')))のエリア・広域ブロック情報/ピーク時供給力(翌々日)は18:00公開のため未反映" Yellow}

 Run-Python '6 DAY1-DAY7予測' (Join-Path $Learning 'predict_day1_to_day7_v1_1.py') @('--forecast-date',$Day1.ToString('yyyy-MM-dd'))

 if(!$SkipSystemModel){

  function Step-Opt([string]$Name,[scriptblock]$Body){try{& $Body|Out-Null;return $true}catch{Log "WARN $Name に失敗（続行）: $($_.Exception.Message)" Yellow;return $false}}

  $chainMark=Join-Path $LogDir 'training_chain_last_ok.txt'

  $todayStr=(Get-Date).ToString('yyyy-MM-dd')

  $markVal=if(Test-Path $chainMark){(Get-Content $chainMark -Encoding ASCII|Select-Object -First 1)}else{''}

  if($ForceTrainingChain -or $markVal -ne $todayStr){

   if(Step-Opt '6a 学習チェーン再作成' {Run-Python '6a 学習チェーン再作成' (Join-Path $Learning 'rebuild_training_chain_v8.py')}){

    $nat=Join-Path $Database '統合データ\training_v8_price_regime\jepx_training_v8_price_regime_system_natwf_day1.parquet'

    if((Test-Path $nat) -and (Get-Item $nat).LastWriteTime.Date -eq (Get-Date).Date){Set-Content $chainMark $todayStr -Encoding ASCII}else{Log "WARN 学習チェーン: system_natwf_day1が更新されていません（再作成は未完了とみなします）" Yellow}

   }

  }else{Log "6a 学習チェーン再作成: 本日実施済みのため省略" Gray}

  [void](Step-Opt '6b OCCTO学習ファイル更新' {Run-Python '6b OCCTO学習ファイル更新' (Join-Path $Learning 'update_occto_da8.py')})

  $okw=Step-Opt '6c 将来の気象特徴量' {Run-Python '6c 将来の気象特徴量' (Join-Path $Learning 'build_future_weather_features_system.py') @('--mode','fetch')}

  if($okw){Step-Opt '6c2 予報の保存' {Run-Python '6c2 予報の保存' (Join-Path $Learning 'archive_future_weather.py')}}

  $okp=Step-Opt '6d 将来の価格特徴量' {Run-Python '6d 将来の価格特徴量' (Join-Path $Learning 'build_future_price_features_system.py') @('--mode','fetch','--date',$Day1.ToString('yyyy-MM-dd'))}

  if($okw -and $okp){[void](Step-Opt '6e システム専用予測' {Run-Python '6e システム専用予測' (Join-Path $Learning 'predict_system_day1_occto.py') @('--mode','predict','--date',$Day1.ToString('yyyy-MM-dd'))})}else{Log "WARN システム専用予測を省略（将来の特徴量が作れませんでした）" Yellow}

  $okr=$false

  if($okw -and $okp -and !$SkipSystemReplace){$okr=Step-Opt '6g システムDAY1を専用予測に置換' {Run-Python '6g システムDAY1を専用予測に置換' (Join-Path $Learning 'replace_system_day1.py')}}

  Step-Opt '6f システム予測の実績比較' {Run-Python '6f システム予測の実績比較' (Join-Path $Learning 'compare_system_day1.py')}

  if(!$SkipAreaModel){

   $oka=Step-Opt '6h エリア別の将来特徴量' {Run-Python '6h エリア別の将来特徴量' (Join-Path $Learning 'build_future_area_features.py') @('--mode','fetch','--date',$Day1.ToString('yyyy-MM-dd'))}

   if($oka){Step-Opt '6i エリア別DAY1予測(新ブレンド)' {Run-Python '6i エリア別DAY1予測(新ブレンド)' (Join-Path $Learning 'predict_area_day1_recon.py') @('--mode','predict','--date',$Day1.ToString('yyyy-MM-dd'))}}else{Log "WARN エリア別予測を省略(将来の特徴量が作れませんでした)" Yellow}

   if($oka -and ($ApplyAreaReplace -or !$SkipAreaReplace)){[void](Step-Opt '6k エリア別DAY1を新ブレンドに置換' {Run-Python '6k エリア別DAY1を新ブレンドに置換' (Join-Path $Learning 'replace_area_day1.py')})}

   Step-Opt '6j エリア別の実績比較' {Run-Python '6j エリア別の実績比較' (Join-Path $Learning 'compare_area_day1.py')}

  }

  if($now.Hour -ge 11 -and $now.Hour -lt 18){Log "WARN 11:00-18:00実行: DAY1のOCCTO(翌々日)は18:00公開のため、システム専用予測はOCCTOなしモデルです" Yellow}

 }

 try{Run-Python '7 精度履歴更新' 'P:\エリアプライス2\web\update_day1_to_day7_accuracy_history_web.py';Run-Python '8 精度ダッシュボード生成' 'P:\エリアプライス2\web\build_accuracy_dashboard_web.py'}catch{Log "WARN ダッシュボード更新失敗: $($_.Exception.Message)" Yellow}

 if($script:Warns.Count -gt 0){$w=@($script:Warns|Select-Object -Unique);Log "`n=== 警告まとめ（$($w.Count)件） ===" Cyan;$w|ForEach-Object{Add-Content $Log "  $_" -Encoding UTF8;Write-Host "  $_" -ForegroundColor Yellow}}

 Log "`nV4.7 SUCCESS log=$Log" Green

}catch{Log ($_|Out-String) Red;Log "V4.7 FAILED log=$Log" Red;exit 1}

