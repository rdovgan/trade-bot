#!/bin/bash
# Trade Bot Stop Script

set -e

# Configuration
BOT_DIR="/Users/rdovgan/Desktop/projects/trade-bot"
PID_FILE="$BOT_DIR/trade_bot.pid"
LOG_FILE="$BOT_DIR/logs/stop_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$BOT_DIR/logs"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    log "ERROR: PID file not found. Bot may not be running."
    # Try to find and kill any running instances
    PIDS=$(pgrep -f "python -m trade_bot.main" || true)
    if [ -n "$PIDS" ]; then
        log "Found running bot processes: $PIDS"
        log "Killing them..."
        kill $PIDS
        log "Processes killed"
    else
        log "No running bot processes found"
    fi
    exit 0
fi

# Read PID
BOT_PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$BOT_PID" > /dev/null 2>&1; then
    log "WARNING: Bot process (PID: $BOT_PID) is not running"
    rm "$PID_FILE"
    exit 0
fi

# Send graceful shutdown signal (SIGTERM)
log "Stopping trade bot (PID: $BOT_PID)..."
kill -TERM "$BOT_PID"

# Wait for process to stop (max 30 seconds)
WAIT_TIME=0
while ps -p "$BOT_PID" > /dev/null 2>&1 && [ $WAIT_TIME -lt 30 ]; do
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
    log "Waiting for bot to stop... ($WAIT_TIME/30)"
done

# Force kill if still running
if ps -p "$BOT_PID" > /dev/null 2>&1; then
    log "WARNING: Bot did not stop gracefully, force killing..."
    kill -9 "$BOT_PID"
    sleep 1
fi

# Remove PID file
rm "$PID_FILE"

log "Trade bot stopped successfully"
