@echo off
echo AlphaFin 시작 중...
start "AlphaFin API" cmd /k "python C:\AlphaFin\src\korean\api_server.py"
timeout /t 2 /nobreak >nul
start "n8n" cmd /k "n8n start"
timeout /t 3 /nobreak >nul
start http://localhost:5678
