@echo off
rem Atalho para os comandos de terminal. Exemplo: iq diario
cd /d "%~dp0"
".venv\Scripts\python.exe" -m iq_agent %*
