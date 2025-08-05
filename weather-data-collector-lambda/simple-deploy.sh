#!/bin/bash

# Simple Lambda Deployment Script (IAM roles already created)
set -e

# Configuration
FUNCTION_NAME="weather-data-collector"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying Weather Data Collector Lambda Function${NC}"

# Get user inputs
echo -e "${YELLOW}Please provide the following information:${NC}"

# Get IAM Role ARN
echo -e "${BLUE}1. IAM Role ARN (from AWS Console):${NC}"
echo "   Example: arn:aws:iam::123456789012:role/lambda-execution-role"
read -p "   Role ARN: " ROLE_ARN

if [ -z "$ROLE_ARN" ]; then
    echo -e "${RED}❌ Role ARN is required${NC}"
    exit 1
fi

# Get API Key
echo -e "${BLUE}2. OpenWeatherMap API Key:${NC}"
read -p "   API Key: " OPENWEATHER_API_KEY

if [ -z "$OPENWEATHER_API_KEY" ]; then
    echo -e "${RED}❌ API Key is required${NC}"
    exit 1
fi

# Step 1: Build deployment package
echo -e "${YELLOW}Step 1: Building deployment package...${NC}"
if [ -f "deploy.sh" ]; then
    chmod +x deploy.sh
    ./deploy.sh
else
    echo -e "${RED}❌ deploy.sh not found${NC}"
    exit 1
fi

# Check if ZIP file was created
if [ ! -f "weather-lambda-deployment.zip" ]; then
    echo -e "${RED}❌ Deployment package not created${NC}"
    exit 1
fi

# Step 2: Deploy Lambda function
echo -e "${YELLOW}Step 2: Deploying Lambda function...${NC}"

# Check if function exists
if aws lambda get-function --function-name $FUNCTION_NAME &>/dev/null; then
    echo -e "${GREEN}📝 Updating existing function...${NC}"

    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://weather-lambda-deployment.zip

    # Update function configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --handler lambda_function.lambda_handler \
        --timeout 90 \
        --memory-size 256 \
        --environment Variables="{OPENWEATHER_API_KEY=$OPENWEATHER_API_KEY}"

    echo -e "${GREEN}✅ Function updated successfully!${NC}"

else
    echo -e "${GREEN}🆕 Creating new function...${NC}"

    # Create function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role $ROLE_ARN \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://weather-lambda-deployment.zip \
        --timeout 90 \
        --memory-size 256 \
        --description "Weather data collector for MTA analysis" \
        --environment Variables="{OPENWEATHER_API_KEY=$OPENWEATHER_API_KEY}"

    echo -e "${GREEN}✅ Function created successfully!${NC}"
fi

# Step 3: Test the function
echo -e "${YELLOW}Step 3: Testing the function...${NC}"
echo -e "${BLUE}Running test with sample locations...${NC}"

aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload file://test-event.json \
    response.json

echo -e "${GREEN}📋 Function Response:${NC}"
if command -v jq &> /dev/null; then
    cat response.json | jq .
else
    cat response.json | python -m json.tool
fi

# Step 4: Show function info
echo -e "${YELLOW}Step 4: Function Information${NC}"
aws lambda get-function --function-name $FUNCTION_NAME --query 'Configuration.[FunctionName,Runtime,Handler,Timeout,MemorySize,LastModified]' --output table

# Step 5: Optional scheduling setup
echo ""
echo -e "${YELLOW}Would you like to set up automatic scheduling? (y/n)${NC}"
read -p "Schedule to run every hour? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    RULE_NAME="weather-collector-schedule"

    # Get AWS Account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

    echo -e "${BLUE}Setting up EventBridge schedule...${NC}"

    # Create EventBridge rule
    aws events put-rule \
        --name $RULE_NAME \
        --schedule-expression "rate(1 hour)" \
        --description "Trigger weather data collection every hour"

    # Add Lambda permission for EventBridge
    aws lambda add-permission \
        --function-name $FUNCTION_NAME \
        --statement-id "allow-eventbridge-$(date +%s)" \
        --action "lambda:InvokeFunction" \
        --principal events.amazonaws.com \
        --source-arn "arn:aws:events:$(aws configure get region):$ACCOUNT_ID:rule/$RULE_NAME" 2>/dev/null || echo "Permission might already exist"

    # Add target to rule
    aws events put-targets \
        --rule $RULE_NAME \
        --targets "Id"="1","Arn"="arn:aws:lambda:$(aws configure get region):$ACCOUNT_ID:function:$FUNCTION_NAME"

    echo -e "${GREEN}✅ Scheduled execution set up! Function will run every hour.${NC}"
    echo -e "${BLUE}You can disable this in the AWS Console under EventBridge > Rules${NC}"
fi

echo ""
echo -e "${BLUE}🎉 Deployment Complete!${NC}"
echo ""
echo -e "${GREEN}📋 Summary:${NC}"
echo -e "  Function Name: ${YELLOW}$FUNCTION_NAME${NC}"
echo -e "  Handler: ${YELLOW}lambda_function.lambda_handler${NC}"
echo -e "  Runtime: ${YELLOW}python3.9${NC}"
echo -e "  Timeout: ${YELLOW}90 seconds${NC}"
echo -e "  Memory: ${YELLOW}256 MB${NC}"
echo ""
echo -e "${BLUE}🔗 Useful Commands:${NC}"
echo "• Test function:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{}' test-response.json"
echo ""
echo "• View logs:"
echo "  aws logs describe-log-groups --log-group-name-prefix /aws/lambda/$FUNCTION_NAME"
echo ""
echo "• Update code only:"
echo "  aws lambda update-function-code --function-name $FUNCTION_NAME --zip-file fileb://weather-lambda-deployment.zip"
echo ""
echo -e "${GREEN}✨ Your Lambda function is ready to collect weather data!${NC}"