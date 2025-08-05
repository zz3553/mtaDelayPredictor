#!/bin/bash

# AWS Lambda Function Setup Script
set -e

# Configuration variables
FUNCTION_NAME="weather-data-collector"
RUNTIME="python3.9"
HANDLER="lambda_function.lambda_handler"
TIMEOUT=90
MEMORY_SIZE=256
DESCRIPTION="Weather data collector for MTA analysis"
ZIP_FILE="weather-lambda-deployment.zip"

# IAM Role ARN - REPLACE WITH YOUR ACTUAL ROLE ARN
# Since you created this manually in the console, update this with your actual role ARN
ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT_ID:role/your-lambda-execution-role-name"

# Environment variables
OPENWEATHER_API_KEY="your_openweather_api_key_here"

echo "🚀 Setting up AWS Lambda function: $FUNCTION_NAME"

# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME &>/dev/null; then
    echo "📝 Function exists. Updating code..."

    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://$ZIP_FILE

    # Update function configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --handler $HANDLER \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --description "$DESCRIPTION" \
        --environment Variables="{OPENWEATHER_API_KEY=$OPENWEATHER_API_KEY}"

    echo "✅ Function updated successfully!"

else
    echo "🆕 Creating new function..."

    # Create function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --role $ROLE_ARN \
        --handler $HANDLER \
        --zip-file fileb://$ZIP_FILE \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --description "$DESCRIPTION" \
        --environment Variables="{OPENWEATHER_API_KEY=$OPENWEATHER_API_KEY}"

    echo "✅ Function created successfully!"
fi

# Get function information
echo ""
echo "📋 Function Information:"
aws lambda get-function --function-name $FUNCTION_NAME --query 'Configuration.[FunctionName,Runtime,Handler,Timeout,MemorySize,LastModified]' --output table

echo ""
echo "🎯 Manual setup reminders:"
echo "1. ✅ Update ROLE_ARN in this script with your actual IAM role"
echo "2. ✅ Update OPENWEATHER_API_KEY with your actual API key"
echo "3. ✅ Ensure your IAM role has Secrets Manager permissions"
echo "4. ✅ Test the function with a test event"

echo ""
echo "🧪 Test the function:"
echo "aws lambda invoke --function-name $FUNCTION_NAME --payload '{}' response.json && cat response.json"