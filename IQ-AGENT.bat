@echo off
rem Abre o painel de controle do IQ-AGENT (sem janela de terminal).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" call instalar.bat
start "" ".venv\Scripts\pythonw.exe" -m iq_agent painel
