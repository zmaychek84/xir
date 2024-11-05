@REM Copyright 2024 Advanced Micro Devices, Inc. All rights reserved.

@REM Licensed under the Apache License, Version 2.0 (the "License"); you may not
@REM use this file except in compliance with the License. You may obtain a copy of
@REM the License at

@REM http://www.apache.org/licenses/LICENSE-2.0

@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
@REM WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
@REM License for the specific language governing permissions and limitations under
@REM the License.
@ECHO OFF

SET PREFIX=C:\Xilinx\XRT\ext

IF "%1" == "-clean" (
  GOTO Clean
)

IF "%1" == "-help" (
  GOTO Help
)

IF "%1" == "-debug" (
  GOTO DebugBuild
)

IF "%1" == "-release" (
  GOTO ReleaseBuild
)

IF "%1" == "-all" (
  CALL:DebugBuild
  IF errorlevel 1 (exit /B %errorlevel%)

  CALL:ReleaseBuild
  IF errorlevel 1 (exit /B %errorlevel%)

  goto:EOF
)

IF "%1" == "" (

  CALL:ReleaseBuild
  IF errorlevel 1 (exit /B %errorlevel%)

  GOTO:EOF
)

ECHO Unknown option: %1
GOTO Help


REM --------------------------------------------------------------------------
:Help
ECHO.
ECHO Usage: build.bat [options]
ECHO.
ECHO [-help]                    - List this help
ECHO [-clean]                   - Remove build directories
ECHO [-debug]                   - Creates a debug build
ECHO [-release]                 - Creates a release build
ECHO [-all]                     - Creates a release build and a debug build
ECHO.
GOTO:EOF

REM --------------------------------------------------------------------------
:Clean
IF EXIST WDebug (
  ECHO Removing 'WDebug' directory...
  rmdir /S /Q WDebug
)
IF EXIST WRelease (
  ECHO Removing 'WRelease' directory...
  rmdir /S /Q WRelease
)
GOTO:EOF

REM --------------------------------------------------------------------------
:DebugBuild
ECHO ====================== Windows Debug Build ============================
MKDIR WDebug
PUSHD WDebug

cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH=%PREFIX% -DCMAKE_INSTALL_PREFIX=%PREFIX% -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
IF errorlevel 1 (POPD & exit /B %errorlevel%)

cmake --build . --verbose --config Debug
IF errorlevel 1 (POPD & exit /B %errorlevel%)

cmake --build . --verbose --config Debug --target install
IF errorlevel 1 (POPD & exit /B %errorlevel%)

ECHO ====================== Zipping up Installation Build ============================
cpack -G ZIP -C Debug

ECHO ====================== Creating MSI Archive ============================
cpack -G WIX -C Debug

POPD
GOTO:EOF

REM --------------------------------------------------------------------------
:ReleaseBuild
ECHO ====================== Windows Release Build ============================
MKDIR WRelease
PUSHD WRelease

cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH=%PREFIX% -DCMAKE_INSTALL_PREFIX=%PREFIX% -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
IF errorlevel 1 (POPD & exit /B %errorlevel%)

cmake --build . --verbose --config Release
IF errorlevel 1 (POPD & exit /B %errorlevel%)

cmake --build . --verbose --config Release --target install
IF errorlevel 1 (POPD & exit /B %errorlevel%)

ECHO ====================== Zipping up Installation Build ============================
cpack -G ZIP -C Release

ECHO ====================== Creating MSI Archive ============================
cpack -G WIX -C Release

POPD
GOTO:EOF


