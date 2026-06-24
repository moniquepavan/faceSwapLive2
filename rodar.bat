@echo off
echo Iniciando Face Swap Live...
echo Abrindo http://localhost:5000
echo Pressione Ctrl+C para encerrar.
echo.

start "" http://localhost:5000
venv\Scripts\python app.py
pause
