#!/bin/bash

echo "📦 Creating Lambda Layer for dependencies..."

# Create layer structure
rm -rf layer/
mkdir -p layer/python

# Install dependencies in layer
python3 -m pip install requests sqlalchemy psycopg2-binary nyct-gtfs -t layer/python/

# Clean up layer
cd layer/python/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.md" -delete 2>/dev/null || true
cd ../..

# Create layer ZIP
cd layer/
zip -r ../mta-dependencies-layer.zip .
cd ..

# Publish layer
echo "☁️  Publishing Lambda Layer..."
LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name mta-dependencies \
    --zip-file fileb://mta-dependencies-layer.zip \
    --compatible-runtimes python3.9 python3.8 python3.10 \
    --region us-east-1 \
    --query 'LayerArn' --output text)

echo "✅ Layer created: $LAYER_ARN"

# Update Lambda function to use layer
echo "🔗 Attaching layer to Lambda function..."
aws lambda update-function-configuration \
    --function-name mta-delay-tracker \
    --layers $LAYER_ARN \
    --region us-east-1

echo "✅ Layer attached to function"

# Clean up
rm -rf layer/
rm mta-dependencies-layer.zip

echo "🎉 Now you can deploy just your code without dependencies!"