@echo off
echo ============================================
echo  Face Swap Live - Instalacao
echo ============================================
echo.

:: Verifica Python
py --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado.
    echo Baixe e instale em: https://www.python.org/downloads/
    echo Marque a opcao "Add Python to PATH" durante a instalacao.
    pause & exit /b 1
)

echo [1/3] Criando ambiente virtual...
py -m venv venv
if errorlevel 1 ( echo ERRO ao criar venv. & pause & exit /b 1 )

echo [2/3] Instalando dependencias...
venv\Scripts\pip install --upgrade pip --quiet
venv\Scripts\pip install -r requirements.txt
if errorlevel 1 ( echo ERRO ao instalar dependencias. & pause & exit /b 1 )

echo.
echo [3/3] Baixando modelo de face swap (~280 MB)...
venv\Scripts\python baixar_modelo.py
if errorlevel 1 (
    echo.
    echo AVISO: Download falhou. Tente rodar "baixar_modelo.py" manualmente depois.
)

echo.
echo ============================================
echo  Instalacao concluida!
echo  Execute: rodar.bat
echo ============================================
pause
