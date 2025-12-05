#!/bin/bash

# MongoDB Setup Script for AlgoClash
# This script automates the MongoDB installation and configuration

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          AlgoClash MongoDB Setup Script                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Step 1: Check if MongoDB is installed
echo "Step 1: Checking MongoDB installation..."
if command -v mongosh &> /dev/null; then
    print_success "MongoDB shell (mongosh) is installed"
    mongosh --version
elif command -v mongo &> /dev/null; then
    print_success "MongoDB shell (mongo) is installed"
    mongo --version
else
    print_error "MongoDB is not installed"
    echo ""
    print_info "Please install MongoDB:"
    echo "  Mac:    brew install mongodb-community"
    echo "  Ubuntu: sudo apt install mongodb"
    echo "  Or visit: https://www.mongodb.com/try/download/community"
    exit 1
fi

# Step 2: Check if MongoDB is running
echo ""
echo "Step 2: Checking if MongoDB is running..."
if mongosh --eval "db.version()" mongodb://localhost:27017 &> /dev/null; then
    print_success "MongoDB is running"
else
    print_error "MongoDB is not running"
    print_info "Starting MongoDB..."

    # Try to start MongoDB based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start mongodb-community
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start mongod
    fi

    # Wait and check again
    sleep 2
    if mongosh --eval "db.version()" mongodb://localhost:27017 &> /dev/null; then
        print_success "MongoDB started successfully"
    else
        print_error "Failed to start MongoDB automatically"
        print_info "Please start MongoDB manually"
        exit 1
    fi
fi

# Step 3: Install Python dependencies
echo ""
echo "Step 3: Installing Python dependencies..."
if pip install pymongo &> /dev/null; then
    print_success "pymongo installed"
else
    print_error "Failed to install pymongo"
    exit 1
fi

# Step 4: Check/Create .env file
echo ""
echo "Step 4: Checking environment configuration..."
ENV_FILE="backend/.env"

if [ ! -f "$ENV_FILE" ]; then
    print_info "Creating .env file..."
    cat > "$ENV_FILE" << EOF
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/ai_trader_battlefield
MONGODB_DATABASE=ai_trader_battlefield

# OpenRouter API
OPENROUTER_API_KEY=your_api_key_here
EOF
    print_success ".env file created"
else
    # Check if MongoDB settings exist
    if ! grep -q "MONGODB_URI" "$ENV_FILE"; then
        print_info "Adding MongoDB settings to .env..."
        echo "" >> "$ENV_FILE"
        echo "# MongoDB Configuration" >> "$ENV_FILE"
        echo "MONGODB_URI=mongodb://localhost:27017/ai_trader_battlefield" >> "$ENV_FILE"
        echo "MONGODB_DATABASE=ai_trader_battlefield" >> "$ENV_FILE"
        print_success "MongoDB settings added to .env"
    else
        print_success ".env file already configured"
    fi
fi

# Step 5: Run tests
echo ""
echo "Step 5: Running MongoDB tests..."
if python test_mongodb.py; then
    print_success "All tests passed!"
else
    print_error "Tests failed"
    print_info "Please check the error messages above"
    exit 1
fi

# Step 6: Backup and update app.py (optional)
echo ""
read -p "Do you want to replace app.py with MongoDB version? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "app.py" ]; then
        cp app.py app_original_backup.py
        print_success "Original app.py backed up as app_original_backup.py"
    fi

    cp app_mongodb.py app.py
    print_success "app.py replaced with MongoDB version"
else
    print_info "Skipped app.py replacement. You can manually copy app_mongodb.py to app.py later"
fi

# Success message
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           MongoDB Setup Completed Successfully!         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
print_success "MongoDB is ready to use!"
echo ""
echo "Next steps:"
echo "  1. Start the Flask app: python app.py"
echo "  2. Open the frontend: npm start (in frontend directory)"
echo "  3. Run a simulation and check MongoDB:"
echo "     mongosh mongodb://localhost:27017/ai_trader_battlefield"
echo ""
print_info "View data in MongoDB:"
echo "  db.simulations.find().pretty()"
echo "  db.generations.find().pretty()"
echo ""
print_info "Documentation:"
echo "  - Setup guide: backend/MONGODB_SETUP.md"
echo "  - Implementation: MONGODB_IMPLEMENTATION.md"
echo ""
