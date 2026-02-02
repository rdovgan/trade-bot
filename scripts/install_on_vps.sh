#!/bin/bash
# One-line installer for VPS - Run this directly on your VPS

set -e

echo "==================================="
echo "Trading Bot VPS Installation"
echo "==================================="
echo ""

# Configuration
BOT_USER=$(whoami)
BOT_DIR="$HOME/trade-bot"
GIT_REPO="https://github.com/rdovgan/trade-bot.git"
SERVICE_TYPE="continuous"  # Options: continuous, oneshot

echo "Configuration:"
echo "  User: $BOT_USER"
echo "  Install Path: $BOT_DIR"
echo "  Git Repo: $GIT_REPO"
echo "  Service Type: $SERVICE_TYPE"
echo ""

read -p "Continue with installation? (y/n) " REPLY
if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
    echo "Installation cancelled."
    exit 1
fi

# Step 1: Install system dependencies
echo ""
echo "Step 1/6: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git curl

# Step 2: Clone repository
echo ""
echo "Step 2/6: Cloning repository..."
if [ -d "$BOT_DIR" ]; then
    echo "Directory $BOT_DIR already exists. Updating..."
    cd "$BOT_DIR"
    # Ensure we're on master branch
    git checkout master 2>/dev/null || git checkout -b master
    git pull origin master
else
    git clone "$GIT_REPO" "$BOT_DIR"
    cd "$BOT_DIR"
fi

# Step 3: Setup Python environment
echo ""
echo "Step 3/6: Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip

# Install dependencies from pyproject.toml
if [ -f "pyproject.toml" ]; then
    pip install -e .
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: No pyproject.toml or requirements.txt found"
fi

# Step 4: Create logs directory
echo ""
echo "Step 4/6: Creating logs directory..."
mkdir -p logs

# Step 5: Configure service files
echo ""
echo "Step 5/6: Configuring systemd service..."

if [ "$SERVICE_TYPE" = "continuous" ]; then
    SERVICE_FILE="tradebot.service"
else
    SERVICE_FILE="tradebot-oneshot.service"
fi

# Update service file with correct paths
sed -i "s|User=rdovgan|User=$BOT_USER|g" "$SERVICE_FILE"
sed -i "s|Group=rdovgan|Group=$BOT_USER|g" "$SERVICE_FILE"
sed -i "s|WorkingDirectory=/home/rdovgan/trade-bot|WorkingDirectory=$BOT_DIR|g" "$SERVICE_FILE"
sed -i "s|/home/rdovgan/trade-bot|$BOT_DIR|g" "$SERVICE_FILE"

# Install service
sudo cp "$SERVICE_FILE" /etc/systemd/system/

# Install timer if oneshot
if [ "$SERVICE_TYPE" = "oneshot" ]; then
    sudo cp tradebot.timer /etc/systemd/system/
fi

sudo systemctl daemon-reload

# Step 6: Setup configuration
echo ""
echo "Step 6/6: Configuration setup..."
if [ ! -f "config.yaml" ] && [ -f "config.example.yaml" ]; then
    echo "Copying example config..."
    cp config.example.yaml config.yaml
    echo "⚠️  Please edit config.yaml with your settings"
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Copying example .env..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your API keys"
fi

echo ""
echo "==================================="
echo "✓ Installation Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo ""
if [ "$SERVICE_TYPE" = "continuous" ]; then
    echo "1. Edit configuration:"
    echo "   nano $BOT_DIR/config.yaml"
    echo "   nano $BOT_DIR/.env"
    echo ""
    echo "2. Enable and start service:"
    echo "   sudo systemctl enable tradebot"
    echo "   sudo systemctl start tradebot"
    echo ""
    echo "3. Check status:"
    echo "   sudo systemctl status tradebot"
    echo ""
    echo "4. View logs:"
    echo "   sudo journalctl -u tradebot -f"
else
    echo "1. Edit configuration:"
    echo "   nano $BOT_DIR/config.yaml"
    echo "   nano $BOT_DIR/.env"
    echo ""
    echo "2. Enable and start timer:"
    echo "   sudo systemctl enable tradebot.timer"
    echo "   sudo systemctl start tradebot.timer"
    echo ""
    echo "3. Check timer:"
    echo "   sudo systemctl list-timers"
    echo ""
    echo "4. Run now (manual):"
    echo "   sudo systemctl start tradebot-oneshot"
fi
echo ""
echo "Service will automatically 'git pull' before each run!"
echo ""
