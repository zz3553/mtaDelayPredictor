#!/bin/bash

# Weather Lambda Deployment Script
set -e

echo "🚀 Starting Lambda deployment package creation..."

# Configuration
LAMBDA_FUNCTION_NAME="weather-data-collector"
PACKAGE_DIR="lambda-package"
ZIP_FILE="weather-lambda-deployment.zip"

# Clean up previous builds
echo "🧹 Cleaning up previous builds..."
rm -rf $PACKAGE_DIR
rm -f $ZIP_FILE

# Create package directory
echo "📁 Creating package directory..."
mkdir $PACKAGE_DIR

# Copy Lambda function
echo "📋 Copying lambda_function.py..."
cp lambda_function.py $PACKAGE_DIR/

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt -t $PACKAGE_DIR/ --no-cache-dir

# Remove unnecessary files to reduce package size
echo "🗑️  Removing unnecessary files..."
cd $PACKAGE_DIR

# Remove test files and documentation but keep metadata
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove documentation and examples but KEEP .dist-info for metadata
find . -name "*.md" -delete 2>/dev/null || true
find . -name "*.rst" -delete 2>/dev/null || true
find . -name "*.txt" -not -path "*dist-info*" -delete 2>/dev/null || true
find . -name "examples" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "docs" -type d -exec rm -rf {} + 2>/dev/null || true

cd ..

# Create ZIP file
echo "📦 Creating deployment ZIP file..."
cd $PACKAGE_DIR
zip -r ../$ZIP_FILE . -q
cd ..

# Check final package size
PACKAGE_SIZE=$(du -h $ZIP_FILE | cut -f1)
echo "✅ Deployment package created: $ZIP_FILE (Size: $PACKAGE_SIZE)"

# Display package contents
echo "📋 Package contents:"
unzip -l $ZIP_FILE | head -20

# Provide next steps
echo ""
echo "🎯 Next Steps:"
echo "1. Upload $ZIP_FILE to your Lambda function"
echo "2. Set the handler to: lambda_function.lambda_handler"
echo "3. Configure environment variables:"
echo "   - OPENWEATHER_API_KEY: your_api_key_here"
echo "4. Set execution timeout to at least 60 seconds"
echo "5. Attach appropriate IAM role with Secrets Manager permissions"
echo ""
echo "📝 AWS CLI deployment command:"
echo "aws lambda update-function-code --function-name $LAMBDA_FUNCTION_NAME --zip-file fileb://$ZIP_FILE"

echo ""
echo "✨ Deployment package ready!"