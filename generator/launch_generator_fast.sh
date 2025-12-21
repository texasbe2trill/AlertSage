#!/bin/bash

# Fast Dataset Generation Script (No LLM)
# Usage: ./launch_generator_fast.sh [events] [dataset_name] [--fresh]
# Example: ./launch_generator_fast.sh 300000 enhanced_dataset --fresh

# Parse arguments
EVENTS=""
DATASET_NAME=""
FRESH_START=false

for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_START=true
            ;;
        *)
            if [ -z "$EVENTS" ]; then
                EVENTS=$arg
            elif [ -z "$DATASET_NAME" ]; then
                DATASET_NAME=$arg
            fi
            ;;
    esac
done

# Set defaults
EVENTS=${EVENTS:-300000}
DATASET_NAME=${DATASET_NAME:-"cyber_incidents_simulated"}
START_DATE="2024-01-01"
END_DATE="2024-12-31"
CHUNK_SIZE=1000

# Derived paths
DATA_DIR="../data"
CSV_FILE="${DATA_DIR}/${DATASET_NAME}.csv"
LOG_FILE="${DATA_DIR}/${DATASET_NAME}.log"
CHECKPOINT_FILE="${DATA_DIR}/${DATASET_NAME}_checkpoint.json"
NOHUP_FILE="${DATA_DIR}/nohup_output.log"

echo "🚀 FAST ENHANCED DATASET GENERATION (NO LLM)"
echo "==============================================="
echo "Events to generate: $EVENTS"
echo "Dataset name: $DATASET_NAME"
echo "Date range: $START_DATE to $END_DATE"
echo "Chunk size: $CHUNK_SIZE"
echo "Output file: $CSV_FILE"
echo ""
echo "✨ ENHANCEMENTS ENABLED:"
echo "  • Synonym augmentation (40% of events)"
echo "  • Borderline scenarios (20% of events)"
echo "  • Expanded templates (3-4x variations)"
echo "  • Label noise + typos (realistic data)"
echo "  • 🚫 LLM rewrites DISABLED (faster generation)"
echo ""
if [ "$FRESH_START" = true ]; then
    echo "Mode: 🧹 FRESH START (delete existing files)"
else
    echo "Mode: 🔄 RESUME/CHECKPOINT (keep existing files)"
fi
echo

# Check if already running
EXISTING_PID=$(pgrep -f "generate_cyber_incidents.py.*${DATASET_NAME}")
if [ -n "$EXISTING_PID" ]; then
    if [ "$FRESH_START" = true ]; then
        echo "🛑 Existing process detected (PID: $EXISTING_PID) - killing for fresh start..."
        pkill -f "generate_cyber_incidents.py.*${DATASET_NAME}"
        sleep 2
        
        STILL_RUNNING=$(pgrep -f "generate_cyber_incidents.py.*${DATASET_NAME}")
        if [ -n "$STILL_RUNNING" ]; then
            echo "❌ ERROR: Failed to kill existing process. Force kill with:"
            echo "   kill -9 $STILL_RUNNING"
            exit 1
        else
            echo "   ✅ Process terminated successfully"
        fi
    else
        echo "❌ ERROR: Generation already running (PID: $EXISTING_PID)"
        echo "   Options:"
        echo "   - Use './monitor_generation.sh $DATASET_NAME' to check status"
        echo "   - Kill with: pkill -f generate_cyber_incidents"
        echo "   - Use '--fresh' flag to automatically kill and restart"
        exit 1
    fi
fi

# Check if files exist and handle fresh start vs resume
if [ -f "$CSV_FILE" ] || [ -f "$CHECKPOINT_FILE" ]; then
    if [ "$FRESH_START" = true ]; then
        echo "🧹 FRESH START: Removing existing files..."
        [ -f "$CSV_FILE" ] && rm "$CSV_FILE" && echo "   ✅ Removed $CSV_FILE"
        [ -f "$CHECKPOINT_FILE" ] && rm "$CHECKPOINT_FILE" && echo "   ✅ Removed $CHECKPOINT_FILE"
        [ -f "$LOG_FILE" ] && rm "$LOG_FILE" && echo "   ✅ Removed $LOG_FILE"
        echo "   🚀 Starting completely fresh!"
    else
        echo "⚠️  WARNING: Existing files detected:"
        [ -f "$CSV_FILE" ] && echo "   - $CSV_FILE"
        [ -f "$CHECKPOINT_FILE" ] && echo "   - $CHECKPOINT_FILE"
        echo
        echo "Choose an option:"
        echo "   r) Resume from checkpoint (default)"
        echo "   f) Fresh start (delete existing files)"
        echo "   q) Quit"
        read -p "Choice (r/f/q): " -n 1 -r
        echo
        case $REPLY in
            [Ff])
                echo "🧹 Removing existing files for fresh start..."
                [ -f "$CSV_FILE" ] && rm "$CSV_FILE" && echo "   ✅ Removed $CSV_FILE"
                [ -f "$CHECKPOINT_FILE" ] && rm "$CHECKPOINT_FILE" && echo "   ✅ Removed $CHECKPOINT_FILE"
                [ -f "$LOG_FILE" ] && rm "$LOG_FILE" && echo "   ✅ Removed $LOG_FILE"
                ;;
            [Qq])
                echo "Aborted."
                exit 1
                ;;
            *)
                echo "📋 Resuming from existing checkpoint..."
                ;;
        esac
    fi
else
    echo "🚀 No existing files found - starting fresh!"
fi

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "⚠️  Virtual environment not found at ../venv"
    echo "   Create it with: python3 -m venv ../venv && ../venv/bin/pip install -r ../requirements.txt"
    exit 1
fi

# Create data directory if needed
mkdir -p "$DATA_DIR"

# Launch generation
echo ""
echo "🎯 Launching fast generation (NO LLM)..."
echo "Command: nohup python3 generate_cyber_incidents.py \\"
echo "  --n-events $EVENTS \\"
echo "  --outfile $CSV_FILE \\"
echo "  --start-date $START_DATE \\"
echo "  --end-date $END_DATE \\"
echo "  --chunk-size $CHUNK_SIZE \\"
echo "  --log-file $LOG_FILE \\"
echo "  --checkpoint-file $CHECKPOINT_FILE \\"
echo "  --no-llm \\"
echo "  > $NOHUP_FILE 2>&1 &"
echo

# Use venv Python directly
PYTHON_BIN="../venv/bin/python"

nohup "$PYTHON_BIN" generate_cyber_incidents.py \
  --n-events "$EVENTS" \
  --outfile "$CSV_FILE" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --chunk-size "$CHUNK_SIZE" \
  --log-file "$LOG_FILE" \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --no-llm \
  > "$NOHUP_FILE" 2>&1 &

PID=$!
echo "✅ Fast generation started!"
echo "   Process ID: $PID"
echo "   Dataset: $DATASET_NAME"
echo "   Target: $EVENTS events"
echo "   Estimated time: 10-15 minutes"
echo

# Wait a moment and check if it started successfully
sleep 3
if kill -0 $PID 2>/dev/null; then
    echo "🎉 Process confirmed running"
    echo
    echo "📊 MONITORING COMMANDS:"
    echo "   Check status: ./monitor_generation.sh $DATASET_NAME"
    echo "   Real-time log: tail -f $LOG_FILE"
    echo "   Nohup output: tail -f $NOHUP_FILE"
    echo "   Kill process: pkill -f generate_cyber_incidents"
    echo
    echo "📁 OUTPUT FILES:"
    echo "   Dataset: $CSV_FILE"
    echo "   Logs: $LOG_FILE"  
    echo "   Checkpoint: $CHECKPOINT_FILE"
    echo "   Nohup Output: $NOHUP_FILE"
    echo
    echo "🚀 Generation running in background!"
    echo "   ~$(echo "scale=0; $EVENTS / 500" | bc) chunks will be written"
    echo "   Progress updates in log file"
else
    echo "❌ ERROR: Process failed to start. Check $NOHUP_FILE for details."
    exit 1
fi
