#!/bin/bash

# 改善版トークン長分析を実行
# - experiments_config.pyから自動的にモデルとタスクを取得
# - COMET、chrF、Accuracyを計算
# - 統合された可視化

echo "======================================================================"
echo "Token Length Analysis V2 - Comprehensive Version"
echo "======================================================================"
echo ""
echo "Features:"
echo "  - Auto-loads models and tasks from experiments_config.py"
echo "  - Calculates Accuracy, chrF, and COMET scores"
echo "  - Token ranges: 0-5, 5-10, 10-15, 15-20"
echo "  - Unified visualization"
echo ""

# Log file
logs_dir="logs"
mkdir -p $logs_dir
log_file="$logs_dir/token_length_analysis_v2_$(date +%Y%m%d_%H%M%S).log"

echo "Log file: $log_file"
echo ""
echo "To monitor progress:"
echo "  tail -f $log_file"
echo ""
echo "Starting in background..."
echo ""

# Run the analysis
nohup python -u -m scripts.experiments.token_length_analysis_v2 > $log_file 2>&1 &

pid=$!
echo "Analysis started in background (PID: $pid)"
echo ""
echo "Monitor with:"
echo "  tail -f $log_file"
echo ""
echo "Check progress:"
echo "  ls outputs/results/main/token_length_analysis_v2/*.pkl | wc -l"
echo ""
echo "After completion, visualize with:"
echo "  python -m scripts.experiments.visualize_token_length_unified --experiment-id token_length_analysis_v2"
