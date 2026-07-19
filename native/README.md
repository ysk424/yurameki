# Yurameki native core

The native module owns all numerical hair work: Cosserat integration, collision,
length projection, strand-quality evaluation, and the convergent per-frame
grooming pass. Blender RNA/depsgraph access remains in Python and passes dense
world-space arrays across one binding boundary per operation.

Development build for the Blender-bundled Python:

```powershell
$python = 'C:\Users\azoo\git\build_windows_Release_x64_vc17_Release\bin\5.2\python\bin\python.exe'
$pythonLib = 'C:\Users\azoo\git\blender\lib\windows_x64\python\313\libs\python313.lib'
cmake -S native -B build/native -G 'Visual Studio 17 2022' -A x64 `
  "-DPython3_EXECUTABLE:FILEPATH=$python" `
  "-DPython3_LIBRARY:FILEPATH=$pythonLib"
cmake --build build/native --config Release
```

This writes `_yurameki_native_0_3_0.cp313-win_amd64.pyd` beside the add-on sources for
local testing. It does not build a Blender extension ZIP.
