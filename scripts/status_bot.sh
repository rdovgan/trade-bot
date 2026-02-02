#!/bin/bash
# Trade Bot Status Script

# Configuration
BOT_DIR="/Users/rdovgan/Desktop/projects/trade-bot"
PID_FILE="$BOT_DIR/trade_bot.pid"
LOG_DIR="$BOT_DIR/logs"

echo "=== Trade Bot Status ==="
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING (no PID file)"

    # Check for orphaned processes
    PIDS=$(pgrep -f "python -m trade_bot.main" || true)
    if [ -n "$PIDS" ]; then
        echo "WARNING: Found orphaned bot processes: $PIDS"
        echo "Run: kill $PIDS"
    fi
    exit 1
fi

# Read PID
BOT_PID=$(cat "$PID_FILE")

# Check if process is running
if ps -p "$BOT_PID" > /dev/null 2>&1; then
    echo "Status: RUNNING"
    echo "PID: $BOT_PID"

    # Get process info
    echo ""
    echo "Process Info:"
    ps -p "$BOT_PID" -o pid,ppid,%cpu,%mem,etime,command

    # Show latest log
    echo ""
    echo "Latest Log File:"
    LATEST_LOG=$(ls -t "$LOG_DIR"/bot_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "$LATEST_LOG"
        echo ""
        echo "Last 10 lines:"
        tail -10 "$LATEST_LOG"
    else
        echo "No log files found"
    fi

    exit 0
else
    echo "Status: NOT RUNNING (stale PID file)"
    echo "Stale PID: $BOT_PID"
    echo "Run stop_bot.sh to clean up"
    exit 1
fi
