#!/bin/bash
# Daily Trade Bot Run Script (for cron)
# This script is designed to be run daily via cron

set -e

# Configuration
BOT_DIR="/Users/rdovgan/Desktop/projects/trade-bot"
VENV_DIR="$BOT_DIR/.venv"
LOG_DIR="$BOT_DIR/logs"
PID_FILE="$BOT_DIR/trade_bot.pid"

# Change to bot directory
cd "$BOT_DIR"

# Create log directory
mkdir -p "$LOG_DIR"

# Log file for this run
LOG_FILE="$LOG_DIR/cron_$(date +%Y%m%d_%H%M%S).log"

# Function to log
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "=== Daily Bot Run Started ==="

# Kill any existing instances
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log "Stopping existing bot instance (PID: $OLD_PID)"
        kill -TERM "$OLD_PID" 2>/dev/null || true
        sleep 5
        # Force kill if still running
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            log "Force killing bot (PID: $OLD_PID)"
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
    fi
    rm "$PID_FILE"
fi

# Also check for any orphaned processes
ORPHANED=$(pgrep -f "python -m trade_bot.main" || true)
if [ -n "$ORPHANED" ]; then
    log "Killing orphaned bot processes: $ORPHANED"
    kill -9 $ORPHANED 2>/dev/null || true
fi

# Git pull
log "Pulling latest code from git..."
if git pull origin master >> "$LOG_FILE" 2>&1; then
    log "Git pull successful"
else
    log "WARNING: Git pull failed"
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Optional: Update dependencies
# log "Updating dependencies..."
# pip install -q --upgrade -r requirements.txt >> "$LOG_FILE" 2>&1

# Start bot
log "Starting trade bot..."
python -m trade_bot.main >> "$LOG_FILE" 2>&1 &
BOT_PID=$!

# Save PID
echo $BOT_PID > "$PID_FILE"

log "Bot started with PID: $BOT_PID"
log "=== Daily Bot Run Complete ==="

# Clean up old logs (keep last 30 days)
find "$LOG_DIR" -name "cron_*.log" -mtime +30 -delete 2>/dev/null || true
