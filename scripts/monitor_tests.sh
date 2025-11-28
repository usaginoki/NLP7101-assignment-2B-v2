#!/bin/bash
# Monitor test progress

PID=187796
LOG_FILE="tests/test_output.log"

echo "Monitoring test process (PID: $PID)"
echo "============================================"

while kill -0 $PID 2>/dev/null; do
    clear
    echo "Test Monitor - $(date)"
    echo "============================================"

    # Process info
    echo -e "\n[Process Status]"
    ps -p $PID -o pid,pcpu,pmem,etime,cmd | tail -1

    # GPU info
    echo -e "\n[GPU Status]"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

    # Log file
    echo -e "\n[Log File (last 20 lines)]"
    if [ -s "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE"
    else
        echo "  (No output yet - Python buffering or still initializing)"
    fi

    echo -e "\n============================================"
    echo "Press Ctrl+C to stop monitoring (tests continue running)"
    sleep 5
done

echo -e "\n\nTests completed!"
echo "Final results:"
tail -50 "$LOG_FILE"
