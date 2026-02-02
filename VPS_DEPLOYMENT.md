# VPS Deployment Guide - Trading Bot

This guide covers deploying your trading bot to a Linux VPS with automatic updates.

---

## 🎯 Quick Start (Recommended)

### 1. Prepare Your VPS

SSH into your VPS and run:
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y python3 python3-venv python3-pip git

# Create bot directory
mkdir -p ~/trade-bot
```

### 2. Clone Repository on VPS

```bash
cd ~
git clone https://github.com/yourusername/trade-bot.git
# OR if using private repo with SSH key:
git clone git@github.com:yourusername/trade-bot.git

cd trade-bot
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Bot

```bash
# Copy and edit configuration
cp config.example.yaml config.yaml
nano config.yaml

# Add API keys (if using .env)
nano .env
```

### 5. Install Systemd Service

**Choose ONE approach:**

#### Option A: Continuous Running (Recommended for 24/7 trading)

```bash
# Edit service file with your paths
nano tradebot.service

# Update these lines:
# User=your-username
# WorkingDirectory=/home/your-username/trade-bot
# Environment="PATH=/home/your-username/trade-bot/.venv/bin:..."

# Install service
sudo cp tradebot.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable tradebot
sudo systemctl start tradebot

# Check status
sudo systemctl status tradebot
```

#### Option B: Scheduled Runs (Runs at specific times)

```bash
# Edit service and timer files
nano tradebot-oneshot.service
nano tradebot.timer

# Update user and paths as above

# Install both files
sudo cp tradebot-oneshot.service /etc/systemd/system/
sudo cp tradebot.timer /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start timer
sudo systemctl enable tradebot.timer
sudo systemctl start tradebot.timer

# Check timer status
sudo systemctl list-timers
```

---

## 🔄 How Auto-Update Works

The service automatically runs `git pull origin master` before starting:

```bash
# In the service file:
ExecStartPre=/bin/bash -c 'cd /home/user/trade-bot && git pull origin master'
```

**This means:**
- ✅ Every time service starts, it pulls latest code
- ✅ For continuous mode: updates on restart
- ✅ For scheduled mode: updates before each run
- ⚠️ Make sure `master` branch is stable!

---

## 🎮 Service Management Commands

### Continuous Service:

```bash
# Start
sudo systemctl start tradebot

# Stop
sudo systemctl stop tradebot

# Restart (will git pull and restart)
sudo systemctl restart tradebot

# Status
sudo systemctl status tradebot

# Enable auto-start on boot
sudo systemctl enable tradebot

# Disable auto-start
sudo systemctl disable tradebot

# View logs (live)
sudo journalctl -u tradebot -f

# View logs (last 100 lines)
sudo journalctl -u tradebot -n 100

# View logs (today only)
sudo journalctl -u tradebot --since today
```

### Scheduled Service (Timer):

```bash
# Start timer
sudo systemctl start tradebot.timer

# Stop timer
sudo systemctl stop tradebot.timer

# Enable timer (auto-start on boot)
sudo systemctl enable tradebot.timer

# Check when next run is scheduled
sudo systemctl list-timers tradebot.timer

# Manually trigger a run now
sudo systemctl start tradebot-oneshot

# View timer status
sudo systemctl status tradebot.timer

# View logs from last run
sudo journalctl -u tradebot-oneshot -n 100
```

---

## 📅 Customizing Schedule (Timer Mode)

Edit `tradebot.timer`:

```bash
sudo nano /etc/systemd/system/tradebot.timer
```

**Daily at specific time (UTC):**
```ini
[Timer]
OnCalendar=*-*-* 08:00:00  # 8:00 AM UTC every day
```

**Multiple times per day:**
```ini
[Timer]
OnCalendar=*-*-* 08:00:00  # 8:00 AM
OnCalendar=*-*-* 20:00:00  # 8:00 PM
```

**Every 6 hours:**
```ini
[Timer]
OnCalendar=*-*-* 00,06,12,18:00:00
```

**Weekdays only at 9 AM:**
```ini
[Timer]
OnCalendar=Mon-Fri *-*-* 09:00:00
```

After editing, reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart tradebot.timer
```

---

## 🔍 Monitoring and Logs

### View Live Logs:
```bash
# System logs (recommended)
sudo journalctl -u tradebot -f

# Or bot's own log file
tail -f ~/trade-bot/logs/bot.log
```

### Check Service Health:
```bash
# Service status
sudo systemctl status tradebot

# Check if process is running
ps aux | grep "python -m trade_bot.main"

# Check resource usage
top -p $(pgrep -f "python -m trade_bot.main")
```

### Log Rotation:

Create `/etc/logrotate.d/tradebot`:
```bash
sudo nano /etc/logrotate.d/tradebot
```

Add:
```
/home/rdovgan/trade-bot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    missingok
    create 0644 rdovgan rdovgan
}
```

---

## 🔐 Security Best Practices

### 1. Use Separate User (Recommended)

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash tradebot

# Setup bot as that user
sudo su - tradebot
cd ~
git clone https://github.com/yourusername/trade-bot.git
# ... continue setup
```

Update service file:
```ini
User=tradebot
Group=tradebot
WorkingDirectory=/home/tradebot/trade-bot
```

### 2. Secure API Keys

```bash
# Store secrets in environment file
sudo nano /etc/default/tradebot

# Add:
EXCHANGE_API_KEY=your_key_here
EXCHANGE_SECRET=your_secret_here

# Protect it
sudo chmod 600 /etc/default/tradebot
```

Update service file:
```ini
EnvironmentFile=/etc/default/tradebot
```

### 3. Firewall

```bash
# Only allow SSH (if web interface not needed)
sudo ufw allow 22/tcp
sudo ufw enable
```

### 4. SSH Key Authentication

```bash
# From your local machine
ssh-copy-id user@your-vps-ip

# Then disable password auth on VPS
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

---

## 🚀 Automated Deployment Script

Use the deployment script for easy updates:

```bash
# From your local machine
chmod +x scripts/deploy_vps.sh

# Edit configuration
nano scripts/deploy_vps.sh
# Set: VPS_USER, VPS_HOST, VPS_PATH

# Deploy
./scripts/deploy_vps.sh
```

This will:
1. Upload latest code
2. Update dependencies
3. Restart service
4. Pull latest from git

---

## 🔧 Troubleshooting

### Service won't start:

```bash
# Check detailed error
sudo systemctl status tradebot
sudo journalctl -u tradebot -n 50

# Common issues:
# - Wrong paths in service file
# - Python venv not activated
# - Missing dependencies
# - Permission issues
```

### Git pull fails:

```bash
# Service may fail if git detects conflicts
# Solution: Configure git to auto-stash

ssh your-vps
cd ~/trade-bot
git config pull.rebase false
git config --global user.email "bot@example.com"
git config --global user.name "Trading Bot"

# Or disable git pull in service file:
# Comment out: ExecStartPre=...git pull...
```

### High CPU/Memory usage:

```bash
# Check resource limits
sudo systemctl show tradebot | grep -i limit

# Add limits to service file:
[Service]
CPUQuota=50%
MemoryLimit=1G
```

### Auto-restart too aggressive:

```bash
# Edit service file
sudo nano /etc/systemd/system/tradebot.service

# Adjust:
RestartSec=300  # Wait 5 minutes before restart
StartLimitBurst=3  # Max 3 restarts
StartLimitIntervalSec=600  # Within 10 minutes
```

---

## 📊 Monitoring Dashboard (Optional)

### Simple Status Check Script:

Create `~/check_bot.sh`:
```bash
#!/bin/bash
echo "=== Trading Bot Status ==="
echo ""
echo "Service Status:"
systemctl is-active tradebot
echo ""
echo "Uptime:"
systemctl show tradebot --property=ActiveEnterTimestamp
echo ""
echo "Memory Usage:"
ps -o rss,vsz,cmd -p $(pgrep -f "python -m trade_bot.main") | awk 'NR>1 {print $1/1024 " MB"}'
echo ""
echo "Latest Log:"
journalctl -u tradebot -n 5 --no-pager
```

Run via cron to get daily emails:
```bash
crontab -e

# Add:
0 9 * * * ~/check_bot.sh | mail -s "Bot Status" your@email.com
```

---

## 🔄 Update Workflow

### Manual Update:
```bash
ssh your-vps
cd ~/trade-bot
git pull
sudo systemctl restart tradebot
```

### Automatic Update (via git hook):
```bash
# On VPS, create post-receive hook
cd ~/trade-bot/.git/hooks
nano post-receive

# Add:
#!/bin/bash
sudo systemctl restart tradebot

# Make executable
chmod +x post-receive
```

---

## ✅ Recommended Configuration

For most trading bots:

**Setup:**
- ✅ Continuous running service (Option A)
- ✅ Auto-restart on failure
- ✅ Auto git pull enabled
- ✅ Separate user for security
- ✅ Log rotation configured
- ✅ Monitoring script set up

**Commands to remember:**
```bash
# Restart to get updates
sudo systemctl restart tradebot

# Check logs
sudo journalctl -u tradebot -f

# Stop trading
sudo systemctl stop tradebot
```

---

## 📞 Quick Command Reference

```bash
# Deploy/Update
./scripts/deploy_vps.sh

# Start/Stop
sudo systemctl start tradebot
sudo systemctl stop tradebot
sudo systemctl restart tradebot

# Status/Logs
sudo systemctl status tradebot
sudo journalctl -u tradebot -f

# Enable/Disable
sudo systemctl enable tradebot
sudo systemctl disable tradebot
```

---

**Last Updated:** 2026-02-02
