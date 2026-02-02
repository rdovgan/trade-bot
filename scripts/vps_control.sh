#!/bin/bash
# Remote VPS Control Script for Trading Bot

# Configuration - CUSTOMIZE THESE
VPS_USER="rdovgan"
VPS_HOST="your-vps-ip-or-domain"
SERVICE_NAME="tradebot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function
run_remote() {
    ssh "$VPS_USER@$VPS_HOST" "$@"
}

# Show usage
usage() {
    echo "Usage: $0 {start|stop|restart|status|logs|update|deploy}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the trading bot"
    echo "  stop     - Stop the trading bot"
    echo "  restart  - Restart the trading bot (pulls updates)"
    echo "  status   - Show bot status"
    echo "  logs     - View live logs"
    echo "  update   - Pull latest code and restart"
    echo "  deploy   - Full deployment (code + dependencies)"
    echo ""
    exit 1
}

# Check configuration
if [ "$VPS_HOST" = "your-vps-ip-or-domain" ]; then
    echo -e "${RED}Error: Please edit this script and set VPS_HOST${NC}"
    echo "Edit: nano $0"
    exit 1
fi

# Main logic
case "${1:-}" in
    start)
        echo -e "${GREEN}Starting trading bot on VPS...${NC}"
        run_remote "sudo systemctl start $SERVICE_NAME"
        echo -e "${GREEN}✓ Bot started${NC}"
        ;;

    stop)
        echo -e "${YELLOW}Stopping trading bot on VPS...${NC}"
        run_remote "sudo systemctl stop $SERVICE_NAME"
        echo -e "${GREEN}✓ Bot stopped${NC}"
        ;;

    restart)
        echo -e "${YELLOW}Restarting trading bot on VPS (will pull updates)...${NC}"
        run_remote "sudo systemctl restart $SERVICE_NAME"
        echo -e "${GREEN}✓ Bot restarted${NC}"
        echo ""
        echo "Checking status..."
        sleep 2
        run_remote "sudo systemctl status $SERVICE_NAME --no-pager -l"
        ;;

    status)
        echo -e "${GREEN}Bot status:${NC}"
        echo ""
        run_remote "sudo systemctl status $SERVICE_NAME --no-pager -l"
        echo ""
        echo -e "${GREEN}Last 10 log lines:${NC}"
        run_remote "sudo journalctl -u $SERVICE_NAME -n 10 --no-pager"
        ;;

    logs)
        echo -e "${GREEN}Connecting to live logs (Ctrl+C to exit)...${NC}"
        echo ""
        run_remote "sudo journalctl -u $SERVICE_NAME -f"
        ;;

    update)
        echo -e "${GREEN}Updating bot on VPS...${NC}"
        run_remote "cd ~/trade-bot && git pull origin master"
        echo ""
        echo -e "${YELLOW}Restarting service...${NC}"
        run_remote "sudo systemctl restart $SERVICE_NAME"
        echo -e "${GREEN}✓ Update complete${NC}"
        ;;

    deploy)
        echo -e "${GREEN}Full deployment to VPS...${NC}"
        if [ -f "scripts/deploy_vps.sh" ]; then
            ./scripts/deploy_vps.sh
        else
            echo -e "${RED}Error: deploy_vps.sh not found${NC}"
            exit 1
        fi
        ;;

    *)
        usage
        ;;
esac
