# MTA Delay Predictor
Application to help predict MTA Train Delays based on weather conditions.

https://mtadelaypredictor-qgkwb8yejdoyyrqtyrjhrc.streamlit.app/


Current Architecture

<img width="767" height="1023" alt="image" src="https://github.com/user-attachments/assets/2e7bcda1-4522-434a-a6c6-e78947aebbdc" />


# Deploying to AWS
rm -rf mta-lambda-directory deployment.zip

mkdir package
cp lambda_function.py ./package/
cd package
zip -r deployment.zip .
cd ..

aws lambda update-function-code   --function-name collectRealTimeMtaInfo   --zip-file fileb://deployment.zip


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
