@echo off
echo ============================================
echo  Face Swap Live - Instalacao
echo ============================================
echo.

echo [1/3] Criando ambiente virtual...
py -m venv venv
if errorlevel 1 (
    echo ERRO: Python nao encontrado. Instale Python 3.10+ primeiro.
    pause & exit /b 1
)

echo [2/3] Instalando dependencias...
venv\Scripts\pip install --upgrade pip --quiet
venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERRO: Falha na instalacao das dependencias.
    pause & exit /b 1
)

echo.
echo [3/3] Baixando modelo de face swap (~500 MB)...
echo      Isso pode demorar dependendo da sua internet.
echo.
venv\Scripts\python baixar_modelo.py
if errorlevel 1 (
    echo.
    echo AVISO: Download falhou. Veja instrucoes acima.
    echo        Voce pode rodar "baixar_modelo.py" novamente depois.
)

echo.
echo ============================================
echo  Instalacao concluida!
echo  Execute: rodar.bat
echo ============================================
pause
