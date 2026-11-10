@echo off
setlocal
cd /d "%~dp0"
echo Criando o ambiente Python do IQ-AGENT...
python -m venv .venv || goto :erro
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip || goto :erro
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :erro
echo.
echo Instalacao concluida. Abra o painel com IQ-AGENT.bat
pause
exit /b 0
:erro
echo.
echo A instalacao falhou. Confira se o Python 3.11 ou mais novo esta instalado.
pause
exit /b 1
