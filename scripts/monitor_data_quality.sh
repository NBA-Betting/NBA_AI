#!/bin/bash
# Daily data quality monitoring script
# Run via cron: 0 6 * * * /path/to/monitor_data_quality.sh

# Activate virtual environment
cd /home/jeff/Documents/NBA_AI
source venv/bin/activate

# Get current date for log filename
DATE=$(date +%Y-%m-%d)
REPORT_DIR="data/quality_reports"
mkdir -p $REPORT_DIR

# Run quality check for current season
CURRENT_SEASON="2025-2026"
python -m src.data_quality \
    --season=$CURRENT_SEASON \
    --quick \
    --output="${REPORT_DIR}/quality_${CURRENT_SEASON}_${DATE}.json"

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Quality check completed: ${DATE}"
else
    echo "❌ Quality check failed: ${DATE}"
    # Send alert (email, Slack, etc.)
    # Example: python -m src.utils.send_alert "Data quality check failed"
fi

# Cleanup old reports (keep last 30 days)
find $REPORT_DIR -name "quality_*.json" -mtime +30 -delete

echo "Report saved to: ${REPORT_DIR}/quality_${CURRENT_SEASON}_${DATE}.json"
