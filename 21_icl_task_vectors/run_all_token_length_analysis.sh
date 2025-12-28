#!/bin/bash

# experiments_config.pyに定義されているすべてのモデルとタスクで
# トークン長分析を実行するスクリプト

echo "======================================================================"
echo "Token Length Analysis - Running All Configured Models and Tasks"
echo "======================================================================"
echo ""

# Log file
logs_dir="logs"
mkdir -p $logs_dir
log_file="$logs_dir/token_length_analysis_all_$(date +%Y%m%d_%H%M%S).log"

echo "Starting batch analysis..."
echo ""
echo "This will run analysis for all models and tasks defined in:"
echo "  core/experiments_config.py"
echo ""
echo "Log file: $log_file"
echo ""
echo "To monitor progress in real-time:"
echo "  tail -f $log_file"
echo ""
echo "To stop the process:"
echo "  ps aux | grep run_all_token_length_analysis"
echo "  kill <PID>"
echo ""
echo "Starting in background..."
echo ""

# Run the analysis in background
nohup python -u run_all_token_length_analysis.py > $log_file 2>&1 &

pid=$!
echo "Analysis started in background (PID: $pid)"
echo ""
echo "Monitor with:"
echo "  tail -f $log_file"
echo ""
echo "Check status:"
echo "  ps aux | grep $pid"
echo ""
echo "After completion, visualize results with:"
echo "  python -m scripts.experiments.visualize_token_length_analysis --experiment-id token_length_analysis"
