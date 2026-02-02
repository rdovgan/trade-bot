#!/bin/bash
# Trade Bot Startup Script

set -e

# Configuration
BOT_DIR="/Users/rdovgan/Desktop/projects/trade-bot"
VENV_DIR="$BOT_DIR/.venv"
LOG_DIR="$BOT_DIR/logs"
PID_FILE="$BOT_DIR/trade_bot.pid"
LOG_FILE="$LOG_DIR/bot_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Change to bot directory
cd "$BOT_DIR"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if bot is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log "ERROR: Bot is already running (PID: $OLD_PID)"
        exit 1
    else
        log "WARNING: Stale PID file found, removing it"
        rm "$PID_FILE"
    fi
fi

# Git pull latest changes
log "Pulling latest changes from git..."
if git pull origin master 2>&1 | tee -a "$LOG_FILE"; then
    log "Git pull successful"
else
    log "WARNING: Git pull failed, continuing with current version"
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/update dependencies (optional, comment out if not needed)
# log "Updating dependencies..."
# pip install -q --upgrade -r requirements.txt 2>&1 | tee -a "$LOG_FILE"

# Start the bot
log "Starting trade bot..."
log "Logs will be written to: $LOG_FILE"

# Run bot in background and save PID
nohup python -m trade_bot.main >> "$LOG_FILE" 2>&1 &
BOT_PID=$!

# Save PID to file
echo $BOT_PID > "$PID_FILE"

log "Trade bot started successfully (PID: $BOT_PID)"
log "View logs: tail -f $LOG_FILE"

# Keep only last 30 log files
log "Cleaning up old log files..."
cd "$LOG_DIR"
ls -t bot_*.log 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true

log "Startup complete"
