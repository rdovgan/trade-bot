# Trade Bot Automation Setup Guide

This guide provides instructions for automating your trade bot with two approaches:
1. **Service Approach (Recommended)** - Full service control with auto-restart
2. **Cron Job Approach** - Simple daily execution

---

## 📁 Scripts Overview

All scripts are located in the `scripts/` directory:

- **start_bot.sh** - Start the bot (pulls git, activates venv, runs bot)
- **stop_bot.sh** - Stop the bot gracefully
- **restart_bot.sh** - Restart the bot
- **status_bot.sh** - Check bot status
- **daily_run.sh** - Daily cron job script

---

## Option 1: Service Approach (Recommended) 🎯

This approach uses macOS launchd to run the bot as a managed service.

### Benefits:
- ✅ Start/stop/restart commands
- ✅ Auto-restart on crash
- ✅ Scheduled daily runs (8:00 AM default)
- ✅ System integration
- ✅ Automatic logging

### Setup Instructions:

#### 1. Make scripts executable
```bash
cd /Users/rdovgan/Desktop/projects/trade-bot
chmod +x scripts/*.sh
```

#### 2. Test the scripts manually
```bash
# Start the bot
./scripts/start_bot.sh

# Check status
./scripts/status_bot.sh

# Stop the bot
./scripts/stop_bot.sh
```

#### 3. Install the launchd service
```bash
# Copy the plist file to LaunchAgents directory
cp com.tradebot.agent.plist ~/Library/LaunchAgents/

# Load the service
launchctl load ~/Library/LaunchAgents/com.tradebot.agent.plist
```

#### 4. Service Management Commands

**Start the bot:**
```bash
launchctl start com.tradebot.agent
```

**Stop the bot:**
```bash
launchctl stop com.tradebot.agent
```

**Check if service is loaded:**
```bash
launchctl list | grep tradebot
```

**Unload the service (disable):**
```bash
launchctl unload ~/Library/LaunchAgents/com.tradebot.agent.plist
```

**Reload service after config changes:**
```bash
launchctl unload ~/Library/LaunchAgents/com.tradebot.agent.plist
launchctl load ~/Library/LaunchAgents/com.tradebot.agent.plist
```

#### 5. Customize Schedule

Edit `com.tradebot.agent.plist` and change the `StartCalendarInterval`:

```xml
<!-- Daily at 8:00 AM -->
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>8</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

For multiple times per day, use an array:
```xml
<key>StartCalendarInterval</key>
<array>
    <!-- 8:00 AM -->
    <dict>
        <key>Hour</key>
        <integer>8</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <!-- 8:00 PM -->
    <dict>
        <key>Hour</key>
        <integer>20</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</array>
```

After editing, reload the service.

---

## Option 2: Simple Cron Job Approach 📅

This approach uses cron to run the bot daily.

### Benefits:
- ✅ Simple setup
- ✅ Daily execution at specified time
- ❌ No auto-restart on crash
- ❌ Manual stop/start required

### Setup Instructions:

#### 1. Make the daily script executable
```bash
cd /Users/rdovgan/Desktop/projects/trade-bot
chmod +x scripts/daily_run.sh
```

#### 2. Test the script manually
```bash
./scripts/daily_run.sh
```

#### 3. Edit crontab
```bash
crontab -e
```

#### 4. Add cron entry

**Run daily at 8:00 AM:**
```cron
0 8 * * * /Users/rdovgan/Desktop/projects/trade-bot/scripts/daily_run.sh
```

**Run daily at 8:00 AM and 8:00 PM:**
```cron
0 8,20 * * * /Users/rdovgan/Desktop/projects/trade-bot/scripts/daily_run.sh
```

**Run every 6 hours:**
```cron
0 */6 * * * /Users/rdovgan/Desktop/projects/trade-bot/scripts/daily_run.sh
```

#### 5. Verify crontab
```bash
crontab -l
```

#### 6. Manual Control

**Stop the bot:**
```bash
./scripts/stop_bot.sh
```

**Start the bot manually:**
```bash
./scripts/start_bot.sh
```

**Check status:**
```bash
./scripts/status_bot.sh
```

---

## 📊 Monitoring and Logs

### Log Locations:

- **Bot logs**: `/Users/rdovgan/Desktop/projects/trade-bot/logs/bot_*.log`
- **Cron logs**: `/Users/rdovgan/Desktop/projects/trade-bot/logs/cron_*.log`
- **Launchd stdout**: `/Users/rdovgan/Desktop/projects/trade-bot/logs/launchd_stdout.log`
- **Launchd stderr**: `/Users/rdovgan/Desktop/projects/trade-bot/logs/launchd_stderr.log`

### View Live Logs:

```bash
# View latest bot log in real-time
tail -f logs/bot_*.log | tail -f $(ls -t logs/bot_*.log | head -1)

# View launchd logs
tail -f logs/launchd_stdout.log

# View latest 100 lines
tail -100 logs/bot_*.log | less
```

### Check Status:

```bash
# Check bot status
./scripts/status_bot.sh

# Check if process is running
ps aux | grep "python -m trade_bot.main"

# Check launchd service status
launchctl list | grep tradebot
```

---

## 🔧 Troubleshooting

### Bot won't start:

1. Check if bot is already running:
   ```bash
   ./scripts/status_bot.sh
   ```

2. Check logs for errors:
   ```bash
   tail -50 logs/bot_*.log | tail -f $(ls -t logs/bot_*.log | head -1)
   ```

3. Try manual start:
   ```bash
   ./scripts/stop_bot.sh
   ./scripts/start_bot.sh
   ```

### Launchd service issues:

1. Check service status:
   ```bash
   launchctl list | grep tradebot
   ```

2. Check launchd logs:
   ```bash
   cat logs/launchd_stderr.log
   ```

3. Reload service:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.tradebot.agent.plist
   launchctl load ~/Library/LaunchAgents/com.tradebot.agent.plist
   ```

### Cron job not running:

1. Check cron is enabled on macOS:
   ```bash
   # Grant Terminal full disk access in System Preferences > Security & Privacy
   ```

2. Check crontab:
   ```bash
   crontab -l
   ```

3. Check cron logs:
   ```bash
   tail -f logs/cron_*.log | tail -f $(ls -t logs/cron_*.log | head -1)
   ```

### Multiple instances running:

```bash
# Kill all bot instances
pkill -f "python -m trade_bot.main"

# Or use the stop script
./scripts/stop_bot.sh
```

---

## 🔒 Security Considerations

1. **API Keys**: Ensure your `.env` file or config contains sensitive data and is not committed to git
2. **Permissions**: Scripts should only be readable/executable by your user
3. **Logs**: Contains trading activity, ensure proper permissions on log directory
4. **Auto-updates**: Git pull runs daily - ensure your master branch is stable

---

## 📝 Configuration Customization

### Change git branch:

Edit `scripts/start_bot.sh` and `scripts/daily_run.sh`:
```bash
# Change this line:
git pull origin master

# To your branch:
git pull origin your-branch-name
```

### Disable git auto-pull:

Comment out the git pull lines in the scripts:
```bash
# log "Pulling latest changes from git..."
# git pull origin master 2>&1 | tee -a "$LOG_FILE"
```

### Enable dependency updates:

Uncomment this line in `scripts/start_bot.sh`:
```bash
# Uncomment to enable:
pip install -q --upgrade -r requirements.txt 2>&1 | tee -a "$LOG_FILE"
```

---

## 🎯 Recommended Setup

For production use, we recommend:

1. **Use the Service Approach (Option 1)** - Better control and reliability
2. **Set schedule to match market hours** - Edit plist file
3. **Enable dependency updates** - Uncomment in start_bot.sh
4. **Monitor logs regularly** - Set up log rotation
5. **Test thoroughly** - Run manually first before scheduling

---

## 📞 Quick Command Reference

### Service Approach:
```bash
# Start
launchctl start com.tradebot.agent

# Stop
launchctl stop com.tradebot.agent

# Status
./scripts/status_bot.sh

# Restart
./scripts/restart_bot.sh

# View logs
tail -f logs/bot_*.log | tail -f $(ls -t logs/bot_*.log | head -1)
```

### Manual Control:
```bash
# Start
./scripts/start_bot.sh

# Stop
./scripts/stop_bot.sh

# Status
./scripts/status_bot.sh

# Restart
./scripts/restart_bot.sh
```

---

## 📚 Additional Resources

- [launchd info](https://www.launchd.info/)
- [macOS launchd documentation](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)
- [Cron syntax](https://crontab.guru/)

---

**Last Updated:** 2026-02-01
