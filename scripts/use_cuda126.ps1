$ErrorActionPreference = "Stop"

$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if (-not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) {
    throw "CUDA 12.6 nvcc.exe not found under $CudaHome"
}

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$CudaBin = Join-Path $CudaHome "bin"
$PathParts = $env:Path -split ";" | Where-Object {
    $_ -and
    ($_ -ne $CudaBin) -and
    ($_ -notmatch "NVIDIA GPU Computing Toolkit\\CUDA\\v11\.7\\bin")
}
$env:Path = $CudaBin + ";" + ($PathParts -join ";")

$VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $VsWhere) {
    $VsInstall = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($VsInstall) {
        $VcVars = Join-Path $VsInstall "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $VcVars) {
            $EnvDump = cmd /s /c "`"$VcVars`" >nul && set"
            foreach ($Line in $EnvDump) {
                $Idx = $Line.IndexOf("=")
                if ($Idx -gt 0) {
                    $Name = $Line.Substring(0, $Idx)
                    $Value = $Line.Substring($Idx + 1)
                    Set-Item -Path "Env:$Name" -Value $Value
                }
            }
        }
    }
}

Write-Host "CUDA_HOME=$env:CUDA_HOME"
Write-Host "CUDA_PATH=$env:CUDA_PATH"
Write-Host "nvcc=$((Get-Command nvcc).Source)"
if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    Write-Host "cl=$((Get-Command cl.exe).Source)"
} else {
    Write-Warning "cl.exe was not found on PATH; run from a VS Developer PowerShell if extension builds fail."
}
