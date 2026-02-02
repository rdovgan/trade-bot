#!/bin/bash
# Trade Bot Restart Script

BOT_DIR="/Users/rdovgan/Desktop/projects/trade-bot"

echo "=== Restarting Trade Bot ==="
echo ""

# Stop the bot
echo "Stopping bot..."
bash "$BOT_DIR/scripts/stop_bot.sh"

# Wait a moment
echo ""
echo "Waiting 3 seconds..."
sleep 3

# Start the bot
echo ""
echo "Starting bot..."
bash "$BOT_DIR/scripts/start_bot.sh"

echo ""
echo "=== Restart Complete ==="
