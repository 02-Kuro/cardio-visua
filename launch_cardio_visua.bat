@echo off
echo =============================
echo Lancement de Cardio-Visua...
echo =============================
start cmd /k "streamlit run app.py"
timeout /t 5
start cmd /k "ngrok http 8501"



