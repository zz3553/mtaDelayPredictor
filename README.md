# MTA Delay Predictor
Application to help predict MTA Train Delays based on weather conditions.

https://mtadelaypredictor-qgkwb8yejdoyyrqtyrjhrc.streamlit.app/


# Interact with the db
`psql -U postgres -d train_delays`

List all tables
`dt`

Checking logs for cron jobs
`cat /Users/mitchel/run_weather_script.sh`
`cat /Users/mitchel/run_mta_script.sh`

Converting jupyer notebooks to python scripts
`jupyter nbconvert --to script /Users/mitchel/Desktop/beep/mtaDelayPredictor/collect_real_time_mta_info.ipynb`

Set execute permissions on the new .py file
`chmod +x /Users/mitchel/Desktop/beep/mtaDelayPredictor/collect_real_time_mta_info.py`

Test the script manually
`/Users/mitchel/run_weather_script.sh`