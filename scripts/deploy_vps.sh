#!/bin/bash
# VPS Deployment Script for Trading Bot

set -e

echo "=== Trading Bot VPS Deployment ==="
echo ""

# Configuration - CUSTOMIZE THESE
VPS_USER="rdovgan"
VPS_HOST="your-vps-ip-or-domain"
VPS_PATH="/home/rdovgan/trade-bot"
SERVICE_TYPE="continuous"  # Options: continuous, oneshot

echo "Configuration:"
echo "  VPS User: $VPS_USER"
echo "  VPS Host: $VPS_HOST"
echo "  VPS Path: $VPS_PATH"
echo "  Service Type: $SERVICE_TYPE"
echo ""

read -p "Is this configuration correct? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Edit this script to update configuration"
    exit 1
fi

# Test SSH connection
echo "1. Testing SSH connection..."
if ssh "$VPS_USER@$VPS_HOST" "echo 'Connection successful'"; then
    echo "   ✓ SSH connection OK"
else
    echo "   ✗ SSH connection failed!"
    echo "   Make sure you can SSH without password (use ssh-copy-id)"
    exit 1
fi
echo ""

# Create directory on VPS
echo "2. Creating directory on VPS..."
ssh "$VPS_USER@$VPS_HOST" "mkdir -p $VPS_PATH/logs"
echo "   ✓ Directory created"
echo ""

# Copy project files (excluding venv, logs, cache)
echo "3. Uploading project files..."
rsync -avz --progress \
    --exclude='.venv' \
    --exclude='logs/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.db' \
    --exclude='*.log' \
    . "$VPS_USER@$VPS_HOST:$VPS_PATH/"
echo "   ✓ Files uploaded"
echo ""

# Setup Python environment on VPS
echo "4. Setting up Python environment on VPS..."
ssh "$VPS_USER@$VPS_HOST" << 'ENDSSH'
set -e
cd /home/rdovgan/trade-bot

# Install Python 3 and venv if not present
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Python environment ready"
ENDSSH
echo "   ✓ Environment setup complete"
echo ""

# Install systemd service
echo "5. Installing systemd service..."
if [ "$SERVICE_TYPE" = "oneshot" ]; then
    SERVICE_FILE="tradebot-oneshot.service"
    TIMER_FILE="tradebot.timer"
else
    SERVICE_FILE="tradebot.service"
fi

# Copy service file
scp "$SERVICE_FILE" "$VPS_USER@$VPS_HOST:/tmp/"

ssh "$VPS_USER@$VPS_HOST" << ENDSSH
set -e

# Install service
sudo mv "/tmp/$SERVICE_FILE" /etc/systemd/system/

# Install timer if oneshot
if [ "$SERVICE_TYPE" = "oneshot" ]; then
    # Copy timer file
    exit 0
fi

# Reload systemd
sudo systemctl daemon-reload

echo "Service installed"
ENDSSH

# Copy timer if oneshot
if [ "$SERVICE_TYPE" = "oneshot" ]; then
    scp "$TIMER_FILE" "$VPS_USER@$VPS_HOST:/tmp/"
    ssh "$VPS_USER@$VPS_HOST" << 'ENDSSH'
sudo mv /tmp/tradebot.timer /etc/systemd/system/
sudo systemctl daemon-reload
ENDSSH
fi

echo "   ✓ Service installed"
echo ""

echo "=== Deployment Complete! ==="
echo ""
echo "Next steps on VPS:"
echo ""
if [ "$SERVICE_TYPE" = "oneshot" ]; then
    echo "  Enable timer:  sudo systemctl enable tradebot.timer"
    echo "  Start timer:   sudo systemctl start tradebot.timer"
    echo "  Check timer:   sudo systemctl list-timers"
    echo "  Run now:       sudo systemctl start tradebot-oneshot"
else
    echo "  Enable service: sudo systemctl enable tradebot"
    echo "  Start service:  sudo systemctl start tradebot"
    echo "  Check status:   sudo systemctl status tradebot"
fi
echo ""
echo "  View logs:      journalctl -u tradebot -f"
echo "  Stop service:   sudo systemctl stop tradebot"
echo ""
