@echo off
echo Iniciando Face Swap Live...
echo.

:: Aguarda 2s e abre o navegador automaticamente
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

venv\Scripts\python app.py
pause
