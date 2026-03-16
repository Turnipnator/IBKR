#!/bin/bash
# Screener cron entrypoint - runs screener at 14:00 UTC Mon-Fri
# Uses a simple sleep loop since Alpine/slim images don't have cron by default

echo "Screener cron started. Will run at 14:00 UTC on weekdays."

while true; do
    # Get current UTC time
    HOUR=$(date -u +%H)
    MIN=$(date -u +%M)
    DOW=$(date -u +%u)  # 1=Mon, 7=Sun

    # Run at 14:00 UTC, Monday-Friday
    if [ "$HOUR" = "14" ] && [ "$MIN" = "00" ] && [ "$DOW" -le 5 ]; then
        echo "$(date -u) - Running daily screener..."
        cd /app && python -m src.screener 2>&1
        echo "$(date -u) - Screener finished with exit code $?"
        # Sleep 61s to avoid re-triggering in the same minute
        sleep 61
    else
        # Check every 30 seconds
        sleep 30
    fi
done
