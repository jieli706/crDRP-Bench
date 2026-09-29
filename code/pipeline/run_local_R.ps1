param([Parameter(Mandatory=$true)][string]$Script, [string[]]$ScriptArgs=@())
$analysisRoot = Split-Path -Parent $PSScriptRoot
$env:LC_ALL = 'English_United States.utf8'
$env:LANG = 'English_United States.utf8'
$env:LANGUAGE = 'en'
$env:R_USER = Join-Path $analysisRoot 'runtime/r-user'
$env:R_LIBS_USER = Join-Path $analysisRoot 'runtime/r-library'
$env:TEMP = Join-Path $analysisRoot 'runtime/tmp'
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
& (Join-Path $analysisRoot 'runtime/R-4.5.2-portable/app/bin/Rscript.exe') --vanilla $Script @ScriptArgs
exit $LASTEXITCODE
