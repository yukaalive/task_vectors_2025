#!/bin/bash

# トークン長分析を実行するラッパースクリプト
# 使用例:
#   ./run_token_length_analysis.sh gemma 2b translation_ja_en_jesc
#   ./run_token_length_analysis.sh gemma 2b translation_ja_en_jesc translation_en_ja_jesc

if [ "$#" -lt 3 ]; then
    echo "Usage: ./run_token_length_analysis.sh <model_type> <model_variant> <task_name> [task_name2 ...]"
    echo ""
    echo "Examples:"
    echo "  ./run_token_length_analysis.sh gemma 2b translation_ja_en_jesc"
    echo "  ./run_token_length_analysis.sh gemma 2b translation_ja_en_jesc translation_en_ja_jesc"
    echo ""
    echo "Available tasks:"
    echo "  - translation_ja_en_jesc"
    echo "  - translation_en_ja_jesc"
    echo "  - translation_ja_en_easy"
    echo "  - translation_en_ja_easy"
    echo "  - translation_ja_en_single"
    echo "  - translation_en_ja_single"
    echo "  - translation_fr_en"
    echo "  - translation_en_fr"
    echo "  - translation_it_en"
    echo "  - translation_en_it"
    echo "  - translation_es_en"
    echo "  - translation_en_es"
    exit 1
fi

# Log file
logs_dir="logs"
mkdir -p $logs_dir
log_file="$logs_dir/token_length_analysis_$1_$2_$(date +%Y%m%d_%H%M%S).log"

echo "Starting token length analysis..."
echo "Model: $1 $2"
echo "Tasks: ${@:3}"
echo ""
echo "Log file: $log_file"
echo ""
echo "Watch the log with:"
echo "  tail -f $log_file"
echo ""

# Run the analysis
nohup python -u -m scripts.experiments.token_length_analysis "$@" > $log_file 2>&1 &

echo "Analysis started in background (PID: $!)"
echo ""
echo "To visualize results after completion:"
echo "  python -m scripts.experiments.visualize_token_length_analysis --experiment-id token_length_analysis"
