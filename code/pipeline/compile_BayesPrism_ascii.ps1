$analysisRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (Test-Path -LiteralPath 'T:\') { throw 'T: is already in use; no drive mapping changed.' }
& subst.exe T: $analysisRoot
if ($LASTEXITCODE -ne 0) { throw 'Temporary analysis drive mapping failed' }
try {
    Push-Location -LiteralPath 'T:\'
    $env:ANALYSIS_ROOT='T:'
    $env:R_USER='T:/runtime/r-user'
    $env:R_LIBS_USER='T:/runtime/r-library'
    $env:TEMP='T:/runtime/tmp'; $env:TMP=$env:TEMP
    $env:LC_ALL='English_United States.utf8';$env:LANG='English_United States.utf8';$env:LANGUAGE='en'
    $env:OMP_NUM_THREADS='2';$env:OPENBLAS_NUM_THREADS='2'
    & 'T:/runtime/R-4.5.2-portable/app/bin/Rscript.exe' --vanilla 'T:/code/install_BayesPrism_local.R' *> (Join-Path $analysisRoot ('runtime/installers/BayesPrism_install_ascii_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log'))
    $analysisExitCode=$LASTEXITCODE
} finally {
    Pop-Location
    & subst.exe T: /D
}
exit $analysisExitCode
