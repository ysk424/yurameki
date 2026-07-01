$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $here "build"

if (!(Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc was not found in PATH. Install CUDA Toolkit or open a CUDA-enabled developer shell."
}

if (!(Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($install) {
            $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) {
                cmd /c "`"$vcvars`" && cd /d `"$here`" && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release"
                exit $LASTEXITCODE
            }
        }
    }
    throw "cl.exe was not found. Run this script from 'x64 Native Tools Command Prompt for VS 2022'."
}

cmake -S $here -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build $build --config Release
