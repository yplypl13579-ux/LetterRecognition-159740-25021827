param([string]$QtRoot='C:\Qt')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskQmake=Get-ChildItem -LiteralPath $QtRoot -Filter qmake.exe -Recurse | Where-Object { $_.FullName -match 'mingw_64\\bin\\qmake.exe$' } | Sort-Object FullName -Descending | Select-Object -First 1
$taskMake=Get-ChildItem -LiteralPath (Join-Path $QtRoot 'Tools') -Filter mingw32-make.exe -Recurse | Sort-Object FullName -Descending | Select-Object -First 1
if(!$taskQmake -or !$taskMake){throw 'Qt MinGW kit not found. Use Qt Creator or pass -QtRoot.'}
$env:PATH=$taskQmake.DirectoryName+';'+$taskMake.DirectoryName+';'+$env:PATH
& $taskMake.FullName gui
if($LASTEXITCODE -ne 0){throw 'GUI build failed.'}
& $taskMake.FullName test
if($LASTEXITCODE -ne 0){throw 'Engine tests failed.'}
Write-Host 'Build and tests passed. Open the .pro in Qt Creator and set working directory to this project folder.'
