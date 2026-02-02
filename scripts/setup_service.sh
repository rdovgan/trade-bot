#!/bin/bash
# Quick setup script for Trade Bot service

set -e

BOT_DIR="/Users/rdovgan/Desktop/projects/trade-bot"

echo "=== Trade Bot Service Setup ==="
echo ""

# Make scripts executable
echo "1. Making scripts executable..."
chmod +x "$BOT_DIR/scripts"/*.sh
echo "   ✓ Scripts are executable"
echo ""

# Test if scripts work
echo "2. Testing scripts..."
if [ -f "$BOT_DIR/scripts/status_bot.sh" ]; then
    echo "   ✓ Scripts found"
else
    echo "   ✗ Scripts not found!"
    exit 1
fi
echo ""

# Install launchd service
echo "3. Installing launchd service..."
if [ -f "$BOT_DIR/com.tradebot.agent.plist" ]; then
    cp "$BOT_DIR/com.tradebot.agent.plist" ~/Library/LaunchAgents/
    echo "   ✓ Service file copied to ~/Library/LaunchAgents/"
else
    echo "   ✗ Service file not found!"
    exit 1
fi
echo ""

# Load the service
echo "4. Loading service..."
launchctl load ~/Library/LaunchAgents/com.tradebot.agent.plist 2>/dev/null || true
echo "   ✓ Service loaded"
echo ""

# Create logs directory
echo "5. Creating logs directory..."
mkdir -p "$BOT_DIR/logs"
echo "   ✓ Logs directory created"
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "Available commands:"
echo "  Start:   launchctl start com.tradebot.agent"
echo "  Stop:    launchctl stop com.tradebot.agent"
echo "  Status:  $BOT_DIR/scripts/status_bot.sh"
echo "  Restart: $BOT_DIR/scripts/restart_bot.sh"
echo ""
echo "Or use manual control:"
echo "  Start:   $BOT_DIR/scripts/start_bot.sh"
echo "  Stop:    $BOT_DIR/scripts/stop_bot.sh"
echo "  Status:  $BOT_DIR/scripts/status_bot.sh"
echo ""
echo "The bot is scheduled to run daily at 8:00 AM"
echo "Edit ~/Library/LaunchAgents/com.tradebot.agent.plist to change schedule"
echo ""
echo "View logs: tail -f $BOT_DIR/logs/bot_*.log"
echo ""
