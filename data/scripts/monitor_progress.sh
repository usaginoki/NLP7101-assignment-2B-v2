#!/bin/bash

# Monitor feature extraction progress

echo "=== Feature Extraction Progress Monitor ==="
echo ""

while true; do
    clear
    echo "=== Feature Extraction Progress Monitor ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check if processes are running
    COMPLEXITY_PID=$(ps aux | grep "complexity_features_parallel.py" | grep -v grep | head -1 | awk '{print $2}')
    LEXICAL_PID=$(ps aux | grep "lexical_features_parallel.py" | grep -v grep | head -1 | awk '{print $2}')

    if [ -z "$COMPLEXITY_PID" ] && [ -z "$LEXICAL_PID" ]; then
        echo "✓ All processes completed!"
        echo ""
        echo "=== Final Results ==="
        echo ""
        echo "Complexity Features:"
        tail -10 /tmp/complexity_parallel.log
        echo ""
        echo "Lexical Features:"
        tail -10 /tmp/lexical_parallel.log
        break
    fi

    # Show complexity progress
    echo "--- Complexity Feature Extraction ---"
    if [ ! -z "$COMPLEXITY_PID" ]; then
        echo "Status: RUNNING (PID: $COMPLEXITY_PID)"
        tail -1 /tmp/complexity_parallel.log | grep "Processing code samples"
    else
        echo "Status: COMPLETED"
    fi
    echo ""

    # Show lexical progress
    echo "--- Lexical Feature Extraction ---"
    if [ ! -z "$LEXICAL_PID" ]; then
        echo "Status: RUNNING (PID: $LEXICAL_PID)"
        tail -1 /tmp/lexical_parallel.log | grep "Processing code samples"
    else
        echo "Status: COMPLETED"
    fi
    echo ""

    # Show CPU and memory usage
    echo "--- System Resources ---"
    echo "Memory Usage:"
    free -h | grep "Mem:"
    echo ""
    echo "Top CPU Processes:"
    ps aux --sort=-%cpu | grep python3 | grep -E "(complexity|lexical)" | head -5 | awk '{printf "  PID %s: %s%% CPU, %s%% MEM\n", $2, $3, $4}'
    echo ""

    # Show file sizes
    echo "--- Output Files ---"
    if [ -f "data/reports/train_complexity_features.csv" ]; then
        echo "train_complexity_features.csv: $(ls -lh data/reports/train_complexity_features.csv 2>/dev/null | awk '{print $5}')"
    fi
    if [ -f "data/reports/train_lexical_features.csv" ]; then
        echo "train_lexical_features.csv: $(ls -lh data/reports/train_lexical_features.csv 2>/dev/null | awk '{print $5}')"
    fi
    if [ -f "data/reports/validation_complexity_features.csv" ]; then
        echo "validation_complexity_features.csv: $(ls -lh data/reports/validation_complexity_features.csv 2>/dev/null | awk '{print $5}')"
    fi
    if [ -f "data/reports/validation_lexical_features.csv" ]; then
        echo "validation_lexical_features.csv: $(ls -lh data/reports/validation_lexical_features.csv 2>/dev/null | awk '{print $5}')"
    fi

    echo ""
    echo "Press Ctrl+C to exit monitor (processes will continue running)"
    echo "Refreshing in 10 seconds..."

    sleep 10
done
