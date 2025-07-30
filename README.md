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


# Deploying to AWS
rm -rf mta-lambda-directory deployment.zip

mkdir package
cp lambda_function.py ./package/
cd package
zip -r ../deployment.zip .
cd ..

aws lambda update-function-code   --function-name collectRealTimeMtaInfo   --zip-file fileb://deployment_package.zip


docker run --rm --entrypoint="" -v $(pwd):/var/task public.ecr.aws/lambda/python:3.10   /bin/bash -c "
    echo 'Installing system packages...'
    yum update -y && yum install -y zip
    echo 'Creating package directory...'
    mkdir -p /tmp/package
    echo 'Installing Python dependencies...'
    pip install --upgrade pip
    pip install -r requirements.txt --target /tmp/package
    echo 'Copying your code...'
    cp *.py /tmp/package/
    echo 'Creating zip file...'
    cd /tmp/package && zip -r /var/task/lambda-deployment.zip .
    echo 'Build complete!'
  "
