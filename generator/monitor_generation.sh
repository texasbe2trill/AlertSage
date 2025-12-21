#!/bin/bash

# Monitor script for cybersecurity dataset generation
# Usage: ./monitor_generation.sh [dataset_name] [--watch interval]

# Cache GPU info to avoid slow system_profiler calls on every run
# Use a stable cache filename (not process-specific) so all invocations can share it
GPU_INFO_CACHE="/tmp/nlp_triage_gpu_cache"
if [[ "$(uname -m)" == "arm64" ]] && [ ! -f "$GPU_INFO_CACHE" ]; then
    {
        system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model:" | head -1 | sed 's/.*: //' | xargs || echo "Apple Silicon GPU"
        system_profiler SPDisplaysDataType 2>/dev/null | grep -i "Total Number of Cores" | awk '{print $NF}' || echo ""
    } > "$GPU_INFO_CACHE" 2>/dev/null &
fi

# Parse arguments
DATASET_NAME=""
WATCH_MODE=""
WATCH_INTERVAL=30
SIMPLE_MODE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --watch|-w)
            WATCH_MODE="true"
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                WATCH_INTERVAL="$2"
                shift
            fi
            shift
            ;;
        --simple|-s)
            SIMPLE_MODE="true"
            shift
            ;;
        --simple-color)
            SIMPLE_MODE="true"
            FORCE_COLOR="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [dataset_name] [--watch interval]"
            echo ""
            echo "Options:"
            echo "  dataset_name      Name of the dataset (default: cyber_incidents_simulated)"
            echo "  --watch, -w       Auto-refresh mode with optional interval in seconds (default: 30)"
            echo "  --simple, -s      Simple mode - ASCII symbols, no colors"
            echo "  --simple-color    Simple mode - ASCII symbols with colors (best for problematic terminals)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Single run with default dataset"
            echo "  $0 my_dataset                        # Single run with custom dataset"
            echo "  $0 --watch                           # Auto-refresh every 30 seconds"
            echo "  $0 --watch 10                        # Auto-refresh every 10 seconds"
            echo "  $0 --simple --watch                  # Auto-refresh with simple symbols, no colors"
            echo "  $0 --simple-color --watch            # Auto-refresh with simple symbols and colors"
            echo "  $0 my_dataset --watch 60             # Custom dataset, refresh every 60 seconds"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$DATASET_NAME" ]]; then
                DATASET_NAME="$1"
            else
                echo "Error: Multiple dataset names provided"
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default dataset name if not provided
DATASET_NAME=${DATASET_NAME:-"cyber_incidents_simulated"}
DATA_DIR="../data"
CSV_FILE="${DATA_DIR}/${DATASET_NAME}.csv"
LOG_FILE="${DATA_DIR}/${DATASET_NAME}.log"
CHECKPOINT_FILE="${DATA_DIR}/${DATASET_NAME}_checkpoint.json"
NOHUP_FILE="${DATA_DIR}/nohup_output.log"

# Detect if colors should be used
# Disable colors if:
# - Not a terminal (e.g., being watched, piped, etc.) - UNLESS FORCE_COLOR is set
# - TERM is not set or is dumb
# - NO_COLOR environment variable is set
# Also detect problematic emoji rendering environments
FORCE_SIMPLE=""
if [[ "$TERM_PROGRAM" == "tmux" ]] || [[ -n "$SSH_CLIENT" ]] || [[ -n "$SSH_TTY" ]]; then
    # In SSH or tmux environments, be more conservative with Unicode
    if [[ -z "$FORCE_UNICODE" ]]; then
        FORCE_SIMPLE="true"
    fi
fi

if [[ -n "$NO_COLOR" ]] || [[ "$TERM" == "dumb" ]] || { [[ ! -t 1 ]] && [[ -z "$FORCE_COLOR" ]]; } || [[ "$FORCE_SIMPLE" == "true" ]] || [[ "$SIMPLE_MODE" == "true" ]]; then
    # No colors (unless FORCE_COLOR is set for simple-color mode)
    if [[ "$FORCE_COLOR" == "true" ]]; then
        # Simple mode WITH colors
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[1;33m'
        BLUE='\033[0;34m'
        PURPLE='\033[0;35m'
        CYAN='\033[0;36m'
        NC='\033[0m' # No Color
        BOLD='\033[1m'
    else
        # No colors at all
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        PURPLE=''
        CYAN=''
        NC=''
        BOLD=''
    fi
    
    # Simple symbols for compatibility (used in both simple modes)
    CHECKMARK="[OK]"
    CROSS="[X]"
    WARNING="[!]"
    ROCKET="=>"
    REFRESH="~"
    ROBOT="[AI]"
    PACKAGE="##"
    CHART=">>>"
    FOLDER="[]"
    NOTES="--"
    TOOLS=">>"
else
    # Colors for better visibility
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
    BOLD='\033[1m'
    
    # Unicode symbols (work in most modern terminals)
    CHECKMARK="✅"
    CROSS="❌"
    WARNING="⚠️"
    ROCKET="🚀"
    REFRESH="🔄"
    ROBOT="🤖"
    PACKAGE="📦"
    CHART="📈"
    FOLDER="📁"
    NOTES="📝"
    TOOLS="🛠️"
fi

# Function to display the monitor content
display_monitor() {
    # Clear screen for watch mode using ANSI escape instead of clear command
    if [[ -n "$WATCH_MODE" ]]; then
        # Use ANSI escape sequences to clear screen and reset cursor
        printf '\033[2J\033[H'
        echo -e "${BOLD}${CYAN}Auto-refreshing every ${WATCH_INTERVAL}s • Press Ctrl+C to exit${NC}"
        echo
    fi

# Get the actual generation start time (prefer checkpoint, fallback to log file)
GENERATION_START=""
LATEST_START=""

# First try to get from checkpoint file (most reliable for resumed generations)
if [ -f "$CHECKPOINT_FILE" ]; then
    GENERATION_START=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('generation_start_time', ''))" 2>/dev/null || echo "")
    
    # Convert ISO format to local timestamp format and handle timezone
    if [ -n "$GENERATION_START" ]; then
        # Use Python to parse ISO datetime with timezone and convert to local time
        LATEST_START=$(python3 -c "
from datetime import datetime
import sys
try:
    dt = datetime.fromisoformat('$GENERATION_START'.replace('Z', '+00:00'))
    # Convert to local time
    local_dt = dt.astimezone()
    print(local_dt.strftime('%Y-%m-%d %H:%M:%S'))
except:
    sys.exit(1)
" 2>/dev/null || echo "")
    fi
fi

# Fallback to log file if checkpoint doesn't have it
if [ -z "$LATEST_START" ] && [ -f "$LOG_FILE" ]; then
    LATEST_START=$(grep "Starting generation:" "$LOG_FILE" 2>/dev/null | head -1 | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}' || echo "")
fi

# ==============================================================================
# Read progress and timing data ONCE at the start to ensure consistency
# ==============================================================================
NOHUP_PROGRESS=$(tail -20 "$NOHUP_FILE" 2>/dev/null | grep "Generating incidents:" | tail -1)
PROGRESS_PAIR=$(echo "$NOHUP_PROGRESS" | grep -o '[0-9]\+/[0-9]\+' | tail -1)
CURRENT_EVENTS=$(echo "$PROGRESS_PAIR" | cut -d'/' -f1)
TOTAL_EVENTS=$(echo "$PROGRESS_PAIR" | cut -d'/' -f2)

# Get first chunk time (actual generation start, excludes LLM init)
FIRST_CHUNK_TIME=$(grep "Writing chunk 1" "$LOG_FILE" 2>/dev/null | head -1 | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}' || echo "")
if [ -n "$FIRST_CHUNK_TIME" ]; then
    GENERATION_START_TIME=$(date -j -f "%Y-%m-%d %H:%M:%S" "$FIRST_CHUNK_TIME" +%s 2>/dev/null || echo "0")
else
    # Fallback to checkpoint start time if first chunk not found
    GENERATION_START_TIME=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LATEST_START" +%s 2>/dev/null || echo "0")
fi
CURRENT_TIME_SNAPSHOT=$(date +%s)
GENERATION_DURATION=$((CURRENT_TIME_SNAPSHOT - GENERATION_START_TIME))
# ==============================================================================

# Calculate ETA if we have progress data and start time
ETA_STRING=""
if [ -f "$NOHUP_FILE" ] && [ -n "$LATEST_START" ]; then
    if [ -n "$PROGRESS_PAIR" ] && [ $GENERATION_DURATION -gt 60 ]; then  # Need at least 1 minute for reliable ETA
        if [ -n "$CURRENT_EVENTS" ] && [ -n "$TOTAL_EVENTS" ] && [ "$CURRENT_EVENTS" -gt 0 ] && [ "$TOTAL_EVENTS" -gt 0 ]; then
            REMAINING_EVENTS=$((TOTAL_EVENTS - CURRENT_EVENTS))
            RATE=$(echo "scale=3; $CURRENT_EVENTS / $GENERATION_DURATION" | bc 2>/dev/null || echo "0")
            
            if [ "$(echo "$RATE > 0" | bc 2>/dev/null)" = "1" ]; then
                ETA_SECONDS=$(echo "scale=0; $REMAINING_EVENTS / $RATE" | bc 2>/dev/null || echo "0")
                ETA_TIMESTAMP=$(date -d "+${ETA_SECONDS} seconds" "+%a %b %d %H:%M:%S %Z %Y" 2>/dev/null || date -v+"${ETA_SECONDS}"S "+%a %b %d %H:%M:%S %Z %Y" 2>/dev/null || echo "Unknown")
                
                if [ -n "$ETA_TIMESTAMP" ]; then
                    ETA_STRING="${ETA_TIMESTAMP}"
                fi
            fi
        fi
    fi
fi

echo -e "${BOLD}${CYAN}${TOOLS}  CYBERSECURITY DATASET GENERATION MONITOR${NC}"
echo -e "${CYAN}==============================================${NC}"
echo -e "${BOLD}Dataset:${NC} $DATASET_NAME"
if [ -n "$LATEST_START" ]; then
    # Convert to local time format like "Tue Nov 18 21:23:58 CST 2025"
    START_FORMATTED=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LATEST_START" "+%a %b %d %H:%M:%S %Z %Y" 2>/dev/null || echo "$LATEST_START")
    echo -e "${BOLD}Started:${NC} $START_FORMATTED"
else
    echo -e "${BOLD}Started:${NC} Unknown"
fi
if [ -n "$ETA_STRING" ]; then
    echo -e "${BOLD}${YELLOW}ETA:${NC} ${YELLOW}$ETA_STRING${NC}"
fi
echo

# Check if process is running
echo -e "${BOLD}${BLUE}${CHART}  PROCESS STATUS${NC}"
echo -e "${BLUE}-----------------${NC}"
PID=$(pgrep -f "generate_cyber_incidents.py.*${DATASET_NAME}")

if [ -n "$PID" ]; then
    echo -e "${GREEN}${CHECKMARK}  Generation process RUNNING${NC} (PID: $PID)"
    
    # Get system info
    TOTAL_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
    TOTAL_MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    if [ "$TOTAL_MEM_BYTES" -gt 0 ]; then
        TOTAL_MEM_GB=$(echo "scale=1; $TOTAL_MEM_BYTES / 1073741824" | bc 2>/dev/null || echo "Unknown")
    else
        TOTAL_MEM_GB="Unknown"
    fi
    
    # Get process info (macOS compatible)
    PROCESS_INFO=$(ps -p "$PID" -o pid,time,pcpu,pmem,command 2>/dev/null | tail -1)
    if [ -n "$PROCESS_INFO" ]; then
        # Extract CPU and memory usage
        CPU_PERCENT=$(echo "$PROCESS_INFO" | awk '{print $3}')
        MEM_PERCENT=$(echo "$PROCESS_INFO" | awk '{print $4}')
        RUNTIME=$(echo "$PROCESS_INFO" | awk '{print $2}')
        
        # CPU usage with context
        if [ "$TOTAL_CORES" != "Unknown" ]; then
            CPU_PER_CORE=$(echo "scale=1; $CPU_PERCENT / $TOTAL_CORES" | bc 2>/dev/null || echo "N/A")
            echo -e "   ${YELLOW}CPU Usage:${NC} ${CPU_PERCENT}% (${CPU_PER_CORE}% per core, ${TOTAL_CORES} cores total)"
        else
            echo -e "   ${YELLOW}CPU Usage:${NC} ${CPU_PERCENT}%"
        fi
        
        # Memory usage with context
        MEM_BYTES=$(ps -p "$PID" -o rss= 2>/dev/null | awk '{print $1*1024}')
        if [ -n "$MEM_BYTES" ] && [ "$MEM_BYTES" -gt 0 ]; then
            if [ "$MEM_BYTES" -gt 1073741824 ]; then  # > 1GB
                MEM_HUMAN=$(echo "scale=1; $MEM_BYTES / 1073741824" | bc 2>/dev/null)GB
            elif [ "$MEM_BYTES" -gt 1048576 ]; then  # > 1MB
                MEM_HUMAN=$(echo "scale=0; $MEM_BYTES / 1048576" | bc 2>/dev/null)MB
            else
                MEM_HUMAN=$(echo "scale=0; $MEM_BYTES / 1024" | bc 2>/dev/null)KB
            fi
            
            if [ "$TOTAL_MEM_GB" != "Unknown" ]; then
                echo -e "   ${PURPLE}Memory Usage:${NC} ${MEM_PERCENT}% (${MEM_HUMAN} of ${TOTAL_MEM_GB}GB total)"
            else
                echo -e "   ${PURPLE}Memory Usage:${NC} ${MEM_PERCENT}% (${MEM_HUMAN})"
            fi
        else
            echo -e "   ${PURPLE}Memory Usage:${NC} ${MEM_PERCENT}%"
        fi
        
        echo -e "   ${CYAN}Runtime:${NC} $RUNTIME (process uptime)"
        
        # GPU/Metal acceleration detection (Apple Silicon)
        if [[ "$(uname -m)" == "arm64" ]]; then
            # Check if LLM is enabled by looking at nohup output (check beginning of file where init happens)
            USING_LLM=$(grep -E "(Initializing LLM|Loading model)" "$NOHUP_FILE" 2>/dev/null | head -1)
            if [ -n "$USING_LLM" ]; then
                # Read cached GPU info (fast) instead of calling system_profiler
                if [ -f "$GPU_INFO_CACHE" ]; then
                    GPU_MODEL=$(sed -n '1p' "$GPU_INFO_CACHE" 2>/dev/null || echo "Apple Silicon GPU")
                    GPU_CORES=$(sed -n '2p' "$GPU_INFO_CACHE" 2>/dev/null || echo "")
                else
                    # Fallback to direct call if cache doesn't exist yet
                    GPU_MODEL="Apple Silicon GPU"
                    GPU_CORES=""
                fi
                
                # Get LLM model info from nohup
                MODEL_INFO=$(grep "Initializing LLM backend:" "$NOHUP_FILE" 2>/dev/null | tail -1 | sed 's/.*\///' || echo "")
                
                # Check LLM report for rewrite stats if available
                REPORT_FILE="${DATA_DIR}/${DATASET_NAME}_llm_report.json"
                if [ -f "$REPORT_FILE" ]; then
                    REWRITES_APPLIED=$(cat "$REPORT_FILE" 2>/dev/null | jq -r '.rewrites_applied // 0' 2>/dev/null || echo "0")
                    REWRITES_ATTEMPTED=$(cat "$REPORT_FILE" 2>/dev/null | jq -r '.rewrites_attempted // 0' 2>/dev/null || echo "0")
                    REWRITE_SUCCESS_RATE=$(cat "$REPORT_FILE" 2>/dev/null | jq -r '.rewrite_success_rate // 0' 2>/dev/null || echo "0")
                fi
                
                # Display GPU header with cores if available
                if [ -n "$GPU_CORES" ]; then
                    echo -e "   ${GREEN}🎮 GPU Acceleration:${NC} Metal (${GPU_MODEL} - ${GPU_CORES} cores)"
                else
                    echo -e "   ${GREEN}🎮 GPU Acceleration:${NC} Metal (${GPU_MODEL})"
                fi
                
                if [ -n "$MODEL_INFO" ]; then
                    echo -e "   ${GREEN}   LLM Model:${NC} ${MODEL_INFO}"
                fi
                
                # Show LLM activity from report if available
                if [ -n "$REWRITES_APPLIED" ] && [ "$REWRITES_APPLIED" -gt 0 ]; then
                    if [ -n "$CURRENT_EVENTS" ] && [ "$CURRENT_EVENTS" -gt 0 ]; then
                        ENHANCEMENT_RATE=$(echo "scale=1; $REWRITES_APPLIED * 100 / $CURRENT_EVENTS" | bc 2>/dev/null || echo "0")
                        echo -e "   ${GREEN}   LLM Activity:${NC} ${REWRITES_APPLIED}/${REWRITES_ATTEMPTED} rewrites processed"
                        echo -e "   ${GREEN}   Enhancement:${NC} ${ENHANCEMENT_RATE}% of events (${REWRITE_SUCCESS_RATE}% success rate)"
                        
                        # Calculate GPU efficiency metrics
                        if [ -n "$LATEST_START" ]; then
                            START_TIME=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LATEST_START" +%s 2>/dev/null || echo "0")
                            CURRENT_TIME=$(date +%s)
                            DURATION=$((CURRENT_TIME - START_TIME))
                            if [ $DURATION -gt 60 ]; then
                                # Rewrites per hour
                                REWRITES_PER_HOUR=$(echo "scale=1; $REWRITES_APPLIED * 3600 / $DURATION" | bc 2>/dev/null || echo "0")
                                # Average seconds per rewrite
                                AVG_REWRITE_TIME=$(echo "scale=2; $DURATION / $REWRITES_APPLIED" | bc 2>/dev/null || echo "0")
                                # GPU throughput (tokens/sec estimate - 13B model avg ~200 tokens per rewrite)
                                ESTIMATED_TOKENS=$(echo "scale=0; $REWRITES_APPLIED * 200" | bc 2>/dev/null || echo "0")
                                TOKENS_PER_SEC=$(echo "scale=1; $ESTIMATED_TOKENS / $DURATION" | bc 2>/dev/null || echo "0")
                                
                                echo -e "   ${CYAN}   GPU Throughput:${NC} ${REWRITES_PER_HOUR}/hr (${AVG_REWRITE_TIME}s avg per rewrite)"
                                echo -e "   ${CYAN}   Inference Speed:${NC} ~${TOKENS_PER_SEC} tokens/sec"
                            fi
                        fi
                    else
                        echo -e "   ${GREEN}   LLM Activity:${NC} ${REWRITES_APPLIED}/${REWRITES_ATTEMPTED} rewrites (${REWRITE_SUCCESS_RATE}% success)"
                    fi
                else
                    # Show processing status based on configuration
                    REWRITE_PROB=$(grep "export NLP_TRIAGE_LLM_REWRITE_PROB" ../generator/launch_generator.sh 2>/dev/null | grep -o '[0-9.]\+' || echo "30")
                    REWRITE_PROB_PCT=$(echo "scale=0; $REWRITE_PROB * 100" | bc 2>/dev/null || echo "30")
                    echo -e "   ${CYAN}   LLM Status:${NC} Enabled (${REWRITE_PROB_PCT}% rewrite probability)"
                    if [ -n "$CURRENT_EVENTS" ] && [ "$CURRENT_EVENTS" -gt 0 ]; then
                        EXPECTED_REWRITES=$(echo "scale=0; $CURRENT_EVENTS * $REWRITE_PROB" | bc 2>/dev/null || echo "0")
                        echo -e "   ${CYAN}   Expected:${NC} ~${EXPECTED_REWRITES} GPU-enhanced incidents so far"
                        
                        # Calculate expected GPU efficiency if we have runtime
                        EXPECTED_REWRITES_INT=$(echo "$EXPECTED_REWRITES / 1" | bc 2>/dev/null || echo "0")
                        if [ -n "$LATEST_START" ] && [ "$EXPECTED_REWRITES_INT" -gt 0 ]; then
                            START_TIME=$(date -j -f "%Y-%m-%d %H:%M:%S" "$LATEST_START" +%s 2>/dev/null || echo "0")
                            CURRENT_TIME=$(date +%s)
                            DURATION=$((CURRENT_TIME - START_TIME))
                            if [ $DURATION -gt 60 ]; then
                                # Estimated throughput
                                EST_REWRITES_PER_HOUR=$(echo "scale=1; $EXPECTED_REWRITES * 3600 / $DURATION" | bc 2>/dev/null || echo "0")
                                EST_AVG_TIME=$(echo "scale=2; $DURATION / $EXPECTED_REWRITES" | bc 2>/dev/null || echo "0")
                                # Estimated tokens/sec (200 tokens per rewrite avg)
                                EST_TOKENS_PER_SEC=$(echo "scale=1; $EXPECTED_REWRITES * 200 / $DURATION" | bc 2>/dev/null || echo "0")
                                echo -e "   ${CYAN}   GPU Throughput:${NC} ~${EST_REWRITES_PER_HOUR}/hr (${EST_AVG_TIME}s per rewrite)"
                                echo -e "   ${CYAN}   Inference Speed:${NC} ~${EST_TOKENS_PER_SEC} tokens/sec"
                            fi
                        fi
                    fi
                fi
            else
                echo -e "   ${YELLOW}🎮 GPU Acceleration:${NC} Not detected (CPU-only mode)"
            fi
        fi
        
        # Resource efficiency metrics
        if [ -n "$CURRENT_EVENTS" ] && [ "$CURRENT_EVENTS" -gt 0 ] && [ "$(echo "$CPU_PERCENT > 0 && $MEM_PERCENT > 0" | bc 2>/dev/null)" = "1" ]; then
            EVENTS_PER_CPU=$(echo "scale=1; $CURRENT_EVENTS / $CPU_PERCENT" | bc 2>/dev/null || echo "0")
            if [ "$TOTAL_MEM_GB" != "Unknown" ]; then
                USED_MEM_GB=$(echo "scale=1; $TOTAL_MEM_GB * $MEM_PERCENT / 100" | bc 2>/dev/null || echo "0")
                EVENTS_PER_GB=$(echo "scale=0; $CURRENT_EVENTS / $USED_MEM_GB" | bc 2>/dev/null || echo "0")
                echo -e "   ${BOLD}${CYAN}Efficiency:${NC} ${EVENTS_PER_CPU} events/CPU%, ${EVENTS_PER_GB} events/GB"
            else
                echo -e "   ${BOLD}${CYAN}Efficiency:${NC} ${EVENTS_PER_CPU} events/CPU%"
            fi
        fi
    else
        echo "   Process info: Unable to retrieve"
    fi
else
    echo -e "${RED}${CROSS} No generation process found${NC}"
fi
echo

# Progress Status
echo -e "${BOLD}${GREEN}${CHART}  PROGRESS STATUS${NC}"
echo -e "${GREEN}------------------${NC}"

# First, try to get real-time progress from nohup output if process is running
REALTIME_PROGRESS=""
if [ -n "$PID" ] && [ -f "$NOHUP_FILE" ]; then
    NOHUP_PROGRESS=$(tail -20 "$NOHUP_FILE" 2>/dev/null | grep "Generating incidents:" | tail -1)
    if [ -n "$NOHUP_PROGRESS" ]; then
        PROGRESS_PAIR=$(echo "$NOHUP_PROGRESS" | grep -o '[0-9]\+/[0-9]\+' | tail -1)
        if [ -n "$PROGRESS_PAIR" ]; then
            CURRENT_EVENTS=$(echo "$PROGRESS_PAIR" | cut -d'/' -f1)
            TOTAL_EVENTS=$(echo "$PROGRESS_PAIR" | cut -d'/' -f2)
            if [ -n "$CURRENT_EVENTS" ] && [ -n "$TOTAL_EVENTS" ] && [ "$TOTAL_EVENTS" -gt 0 ]; then
                PROGRESS=$(echo "scale=1; $CURRENT_EVENTS * 100 / $TOTAL_EVENTS" | bc 2>/dev/null || echo "0")
                
                                # Create enhanced progress bar with percentage inside
                PROGRESS_INT=$(echo "$PROGRESS / 1" | bc 2>/dev/null || echo "0")
                BAR_LENGTH=50  # Wider bar
                FILLED_LENGTH=$(echo "($PROGRESS_INT * $BAR_LENGTH) / 100" | bc 2>/dev/null || echo "0")
                
                BAR=""
                # shellcheck disable=SC2034
                for i in $(seq 1 "$FILLED_LENGTH" 2>/dev/null); do BAR="${BAR}█"; done
                # shellcheck disable=SC2034
                for i in $(seq $((FILLED_LENGTH + 1)) $BAR_LENGTH 2>/dev/null); do BAR="${BAR}░"; done
                
                echo -e "${BOLD}${ROCKET} Generation active:${NC} ${CYAN}$CURRENT_EVENTS${NC}/${CYAN}$TOTAL_EVENTS${NC} (${BOLD}${YELLOW}${PROGRESS}%${NC})"
                echo -e "   ${GREEN}[$BAR]${NC} ${PROGRESS}% Complete"
                echo -e "   ${BOLD}Status:${NC} Generating events..."
                
                # Throughput trend analysis
                RECENT_SAMPLES=$(tail -200 "$NOHUP_FILE" 2>/dev/null | grep "Generating incidents:" | tail -20 | grep -o '[0-9]\+/[0-9]\+' | cut -d'/' -f1)
                SAMPLE_COUNT=$(echo "$RECENT_SAMPLES" | grep -c '^[0-9]\+$' 2>/dev/null || echo "0")
                
                if [ "$SAMPLE_COUNT" -gt 5 ]; then
                    FIRST_SAMPLE=$(echo "$RECENT_SAMPLES" | head -1)
                    MIDDLE_SAMPLE=$(echo "$RECENT_SAMPLES" | sed -n "$((SAMPLE_COUNT/2))p")
                    LAST_SAMPLE=$(echo "$RECENT_SAMPLES" | tail -1)
                    
                    if [ -n "$FIRST_SAMPLE" ] && [ -n "$MIDDLE_SAMPLE" ] && [ -n "$LAST_SAMPLE" ] && [ "$LAST_SAMPLE" -gt "$FIRST_SAMPLE" ]; then
                        EARLY_RATE=$(echo "scale=3; ($MIDDLE_SAMPLE - $FIRST_SAMPLE) / 300" | bc 2>/dev/null || echo "0")
                        RECENT_RATE=$(echo "scale=3; ($LAST_SAMPLE - $MIDDLE_SAMPLE) / 300" | bc 2>/dev/null || echo "0")
                        
                        if [ "$(echo "$EARLY_RATE > 0 && $RECENT_RATE > 0" | bc 2>/dev/null)" = "1" ]; then
                            TREND_RATIO=$(echo "scale=2; $RECENT_RATE / $EARLY_RATE" | bc 2>/dev/null || echo "1")
                            
                            if [ "$(echo "$TREND_RATIO > 1.1" | bc 2>/dev/null)" = "1" ]; then
                                TREND_PCT=$(echo "scale=0; ($TREND_RATIO - 1) * 100" | bc 2>/dev/null || echo "0")
                                echo -e "   ${GREEN}${CHART} Throughput: Accelerating (+${TREND_PCT}%)${NC}"
                            elif [ "$(echo "$TREND_RATIO < 0.9" | bc 2>/dev/null)" = "1" ]; then
                                TREND_PCT=$(echo "scale=0; (1 - $TREND_RATIO) * 100" | bc 2>/dev/null || echo "0")
                                echo -e "   ${YELLOW}📉 Throughput: Declining (-${TREND_PCT}%)${NC}"
                            else
                                echo -e "   ${CYAN}${CHART} Throughput: Steady${NC}"
                            fi
                        fi
                    fi
                fi
                REALTIME_PROGRESS="true"
            fi
        fi
    fi
fi

# If no real-time progress available, fall back to checkpoint
if [ -z "$REALTIME_PROGRESS" ] && [ -f "$CHECKPOINT_FILE" ]; then
    COMPLETED=$(cat "$CHECKPOINT_FILE" | jq -r '.last_completed_event // 0' 2>/dev/null || echo "0")
    TOTAL=$(cat "$CHECKPOINT_FILE" | jq -r '.total_events // 0' 2>/dev/null || echo "0")
    CHUNKS=$(cat "$CHECKPOINT_FILE" | jq -r '.chunks_written // 0' 2>/dev/null || echo "0")
    UPDATED=$(cat "$CHECKPOINT_FILE" | jq -r '.timestamp // "Unknown"' 2>/dev/null || echo "Unknown")
    STATUS=$(cat "$CHECKPOINT_FILE" | jq -r '.status // "running"' 2>/dev/null || echo "running")
    
    if [ -n "$PID" ] && [ "$STATUS" = "completed" ]; then
        echo -e "${YELLOW}${ROCKET} New generation: Initializing (target: 50000 events)${NC}"
        echo -e "   ${BOLD}Status:${NC} Starting up..."
        echo -e "   ${BOLD}Resume point:${NC} Event $((COMPLETED + 1))"
    else
        if [ "$TOTAL" -gt 0 ]; then
            PROGRESS=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc 2>/dev/null || echo "0")
            echo -e "${GREEN}${CHECKMARK} Events: ${CYAN}$COMPLETED${NC} / ${CYAN}$TOTAL${NC} (${BOLD}${YELLOW}${PROGRESS}%${NC}) [from checkpoint]"
        else
            echo -e "${GREEN}${CHECKMARK} Events: ${CYAN}$COMPLETED${NC} [from checkpoint]"
        fi
        echo -e "   ${BOLD}Chunks written:${NC} $CHUNKS"
        echo -e "   ${BOLD}Status:${NC} $STATUS"
        echo -e "   ${BOLD}Last update:${NC} $UPDATED"
    fi
elif [ -z "$REALTIME_PROGRESS" ]; then
    # No real-time progress and no checkpoint file
    if [ -n "$PID" ]; then
        # Check if LLM is loading
        if [ -f "$NOHUP_FILE" ]; then
            LLM_LOADING=$(tail -20 "$NOHUP_FILE" 2>/dev/null | grep -E "(Initializing LLM|Loading model)" | tail -1)
            if [ -n "$LLM_LOADING" ]; then
                echo -e "${CYAN}${ROBOT} LLM initialization in progress...${NC}"
                echo -e "   ${BOLD}Status:${NC} Loading model for enhanced generation..."
            else
                echo -e "${YELLOW}${REFRESH} Generation starting up...${NC}"
                echo -e "   ${BOLD}Status:${NC} Initializing..."
            fi
        else
            echo -e "${YELLOW}${REFRESH} Generation starting up...${NC}"
            echo -e "   ${BOLD}Status:${NC} Initializing..."
        fi
    else
        echo -e "${YELLOW}${WARNING}  No checkpoint file found${NC}"
    fi
fi
echo

# File Status
echo -e "${BOLD}${PURPLE}${FOLDER}  FILE STATUS${NC}"
echo -e "${PURPLE}--------------${NC}"
if [ -f "$CSV_FILE" ]; then
    SIZE=$(stat -f%z "$CSV_FILE" 2>/dev/null || echo "0")
    if [ "$SIZE" != "0" ]; then
        # Convert to human readable with guaranteed unit suffix
        if [ "$SIZE" -gt 1073741824 ]; then  # > 1GB
            SIZE=$(echo "scale=1; $SIZE / 1073741824" | bc 2>/dev/null)G
        elif [ "$SIZE" -gt 1048576 ]; then  # > 1MB
            SIZE=$(echo "scale=1; $SIZE / 1048576" | bc 2>/dev/null)M
        elif [ "$SIZE" -gt 1024 ]; then  # > 1KB
            SIZE=$(echo "scale=0; $SIZE / 1024" | bc 2>/dev/null)K
        else
            SIZE="${SIZE}B"
        fi
    else
        SIZE="0B"
    fi
    LINES=$(wc -l < "$CSV_FILE" 2>/dev/null || echo "0")
    EVENTS=$((LINES - 1))  # Subtract header
    
    if [ -n "$PID" ] && [ -f "$CHECKPOINT_FILE" ]; then
        CHECKPOINT_STATUS=$(cat "$CHECKPOINT_FILE" | jq -r '.status // "running"' 2>/dev/null || echo "running")
        if [ "$CHECKPOINT_STATUS" = "completed" ]; then
            echo -e "${CYAN}${ROCKET} Dataset: $SIZE (building to 50000 events)${NC}"
            echo -e "${YELLOW}${REFRESH} Current progress: $EVENTS existing + new events being generated${NC}"
        else
            echo -e "${GREEN}${CHECKMARK} Dataset: $SIZE ($EVENTS events)${NC}"
        fi
    else
        echo -e "${GREEN}${CHECKMARK} Dataset: $SIZE ($EVENTS events)${NC}"
    fi
else
    echo -e "${RED}${WARNING}  Dataset file not found: $CSV_FILE${NC}"
fi

if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || echo "0")
    if [ "$LOG_SIZE" != "0" ]; then
        # Convert to human readable with guaranteed unit suffix
        if [ "$LOG_SIZE" -gt 1073741824 ]; then  # > 1GB
            LOG_SIZE=$(echo "scale=1; $LOG_SIZE / 1073741824" | bc 2>/dev/null)G
        elif [ "$LOG_SIZE" -gt 1048576 ]; then  # > 1MB
            LOG_SIZE=$(echo "scale=1; $LOG_SIZE / 1048576" | bc 2>/dev/null)M
        elif [ "$LOG_SIZE" -gt 1024 ]; then  # > 1KB
            LOG_SIZE=$(echo "scale=0; $LOG_SIZE / 1024" | bc 2>/dev/null)K
        else
            LOG_SIZE="${LOG_SIZE}B"
        fi
    else
        LOG_SIZE="0B"
    fi
    echo -e "${GREEN}${CHECKMARK} Log file: $LOG_SIZE${NC}"
else
    echo -e "${RED}${WARNING}  Log file not found: $LOG_FILE${NC}"
fi

if [ -f "$NOHUP_FILE" ]; then
    NOHUP_SIZE=$(stat -f%z "$NOHUP_FILE" 2>/dev/null || echo "0")
    if [ "$NOHUP_SIZE" != "0" ]; then
        # Convert to human readable with guaranteed unit suffix
        if [ "$NOHUP_SIZE" -gt 1073741824 ]; then  # > 1GB
            NOHUP_SIZE=$(echo "scale=1; $NOHUP_SIZE / 1073741824" | bc 2>/dev/null)G
        elif [ "$NOHUP_SIZE" -gt 1048576 ]; then  # > 1MB
            NOHUP_SIZE=$(echo "scale=1; $NOHUP_SIZE / 1048576" | bc 2>/dev/null)M
        elif [ "$NOHUP_SIZE" -gt 1024 ]; then  # > 1KB
            NOHUP_SIZE=$(echo "scale=0; $NOHUP_SIZE / 1024" | bc 2>/dev/null)K
        else
            NOHUP_SIZE="${NOHUP_SIZE}B"
        fi
    else
        NOHUP_SIZE="0B"
    fi
    echo -e "${GREEN}${CHECKMARK} Nohup output: $NOHUP_SIZE${NC}"
fi
echo

# Recent Activity & Chunk Analysis
echo -e "${BOLD}${CYAN}${NOTES}  RECENT ACTIVITY${NC}"
echo -e "${CYAN}------------------${NC}"
if [ -f "$LOG_FILE" ]; then
    echo "Last 5 log entries:"
    tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/   /'
    
    # Chunk timing analysis
    CHUNK_TIMES=$(grep "Writing chunk" "$LOG_FILE" 2>/dev/null | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}')
    CHUNK_COUNT=$(echo "$CHUNK_TIMES" | grep -c '^[0-9]' 2>/dev/null || echo "0")
    
    if [ "$CHUNK_COUNT" -gt 1 ] 2>/dev/null; then
        echo
        echo -e "${BOLD}${PACKAGE} Chunk Analysis:${NC}"
        
        # Calculate time between last few chunks
        RECENT_CHUNKS=$(echo "$CHUNK_TIMES" | tail -5)
        INTERVALS=""
        PREV_TIME=""
        
        while read -r CHUNK_TIME; do
            if [ -n "$PREV_TIME" ] && [ -n "$CHUNK_TIME" ]; then
                PREV_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$PREV_TIME" +%s 2>/dev/null || echo "0")
                CURR_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$CHUNK_TIME" +%s 2>/dev/null || echo "0")
                # Ensure both are valid integers before comparing
                if [ -n "$CURR_EPOCH" ] && [ -n "$PREV_EPOCH" ] && \
                   [ "$CURR_EPOCH" -eq "$CURR_EPOCH" ] 2>/dev/null && \
                   [ "$PREV_EPOCH" -eq "$PREV_EPOCH" ] 2>/dev/null && \
                   [ "$CURR_EPOCH" -gt "$PREV_EPOCH" ] 2>/dev/null; then
                    INTERVAL=$((CURR_EPOCH - PREV_EPOCH))
                    INTERVALS="$INTERVALS $INTERVAL"
                fi
            fi
            PREV_TIME="$CHUNK_TIME"
        done <<< "$RECENT_CHUNKS"
        
        if [ -n "$INTERVALS" ]; then
            # Calculate average interval
            TOTAL_INTERVAL=0
            INTERVAL_COUNT=0
            for INTERVAL in $INTERVALS; do
                TOTAL_INTERVAL=$((TOTAL_INTERVAL + INTERVAL))
                INTERVAL_COUNT=$((INTERVAL_COUNT + 1))
            done
            
            if [ "$INTERVAL_COUNT" -gt 0 ]; then
                AVG_INTERVAL=$((TOTAL_INTERVAL / INTERVAL_COUNT))
                AVG_MINUTES=$((AVG_INTERVAL / 60))
                AVG_SECONDS=$((AVG_INTERVAL % 60))
                
                # Get most recent interval
                LAST_INTERVAL=$(echo "$INTERVALS" | awk '{print $NF}')
                LAST_MINUTES=$((LAST_INTERVAL / 60))
                LAST_SECONDS=$((LAST_INTERVAL % 60))
                
                echo -e "   ${BOLD}Average chunk interval:${NC} ${AVG_MINUTES}m ${AVG_SECONDS}s"
                echo -e "   ${BOLD}Last chunk interval:${NC} ${LAST_MINUTES}m ${LAST_SECONDS}s"
                echo -e "   ${BOLD}Total chunks completed:${NC} $CHUNK_COUNT"
            fi
        fi
    fi
else
    echo "No log file available"
fi
echo

# Performance
if [ -n "$PID" ]; then
    echo -e "${BOLD}${YELLOW}⚡ PERFORMANCE${NC}"
    echo -e "${YELLOW}-------------${NC}"
    
    # Calculate generation runtime from log file
    if [ -f "$LOG_FILE" ] && [ -n "$LATEST_START" ]; then
        # Use cached generation timing values from header calculation
        DURATION=$GENERATION_DURATION
        
        if [ $DURATION -gt 60 ]; then  # More than 1 minute
            # Format duration nicely
            HOURS=$((DURATION / 3600))
            MINUTES=$(( (DURATION % 3600) / 60 ))
            SECONDS=$((DURATION % 60))
                
                if [ $HOURS -gt 0 ]; then
                    DURATION_STR="${HOURS}h ${MINUTES}m ${SECONDS}s"
                elif [ $MINUTES -gt 0 ]; then
                    DURATION_STR="${MINUTES}m ${SECONDS}s"
                else
                    DURATION_STR="${SECONDS}s"
                fi
                
                echo -e "   ${BOLD}Generation runtime:${NC} $DURATION_STR"
                echo -e "   ${BOLD}Started:${NC} $LATEST_START"
                
                # Calculate performance metrics using cached progress data
                if [ -n "$CURRENT_EVENTS" ] && [ "$CURRENT_EVENTS" -gt 0 ]; then
                    # Average rate
                    OVERALL_RATE=$(echo "scale=3; $CURRENT_EVENTS / $DURATION" | bc 2>/dev/null || echo "0")
                    
                    # Time per event (more intuitive than events/second for slow processes)
                    TIME_PER_EVENT=$(echo "scale=1; $DURATION / $CURRENT_EVENTS" | bc 2>/dev/null || echo "0")
                    
                    # Remaining time estimate - use RATE for accuracy (matches ETA header)
                    if [ -n "$TOTAL_EVENTS" ] && [ "$TOTAL_EVENTS" -gt "$CURRENT_EVENTS" ]; then
                        REMAINING_EVENTS=$((TOTAL_EVENTS - CURRENT_EVENTS))
                        # Use rate calculation (same as ETA header)
                        if [ "$(echo "$OVERALL_RATE > 0" | bc 2>/dev/null)" = "1" ]; then
                            REMAINING_SECONDS=$(echo "scale=0; $REMAINING_EVENTS / $OVERALL_RATE" | bc 2>/dev/null || echo "0")
                            # Convert decimal to integer for bash arithmetic
                            REMAINING_SECONDS_INT=$(echo "$REMAINING_SECONDS" | cut -d'.' -f1)
                            REMAINING_HOURS=$((REMAINING_SECONDS_INT / 3600))
                            REMAINING_MINUTES=$(( (REMAINING_SECONDS_INT % 3600) / 60 ))
                            
                            if [ $REMAINING_HOURS -gt 0 ]; then
                                REMAINING_STR="${REMAINING_HOURS}h ${REMAINING_MINUTES}m"
                            elif [ $REMAINING_MINUTES -gt 0 ]; then
                                REMAINING_STR="${REMAINING_MINUTES}m"
                            else
                                REMAINING_STR="<1m remaining"
                            fi
                        fi
                    fi
                    
                    # Progress velocity (% per hour)
                    if [ -n "$TOTAL_EVENTS" ] && [ "$TOTAL_EVENTS" -gt 0 ]; then
                        PERCENT_COMPLETE=$(echo "scale=2; $CURRENT_EVENTS * 100 / $TOTAL_EVENTS" | bc 2>/dev/null || echo "0")
                        HOURS_ELAPSED=$(echo "scale=2; $DURATION / 3600" | bc 2>/dev/null || echo "0")
                        if [ "$(echo "$HOURS_ELAPSED > 0" | bc 2>/dev/null)" = "1" ]; then
                            VELOCITY=$(echo "scale=1; $PERCENT_COMPLETE / $HOURS_ELAPSED" | bc 2>/dev/null || echo "0")
                            echo -e "   ${BOLD}Progress velocity:${NC} ${VELOCITY}%/hour"
                        fi
                    fi
                    
                    echo -e "   ${BOLD}Time per event:${NC} ${TIME_PER_EVENT}s"
                    if [ -n "$REMAINING_STR" ]; then
                        echo -e "   ${BOLD}Estimated time remaining:${NC} $REMAINING_STR"
                    fi
                    echo -e "   ${BOLD}Events/second:${NC} $OVERALL_RATE (avg)"
                fi
            else
                echo -e "   ${BOLD}Generation runtime:${NC} Just started"
            fi
        else
            echo -e "   ${BOLD}Generation runtime:${NC} Unable to determine"
        fi
    else
        echo "   No log file available"
    fi
echo

# Quick Actions
echo -e "${BOLD}${BLUE}${TOOLS}  QUICK ACTIONS${NC}"
echo -e "${BLUE}----------------${NC}"
echo -e "${CYAN}Monitor real-time:${NC} tail -f $LOG_FILE"
echo -e "${CYAN}Check progress:${NC}    cat $CHECKPOINT_FILE | jq"
echo -e "${RED}Kill process:${NC}      pkill -f generate_cyber_incidents"
echo -e "${CYAN}View nohup output:${NC} tail -f $NOHUP_FILE"
echo

# Status Summary
if [ -n "$PID" ]; then
    # First check for real-time progress
    REALTIME_STATUS=""
    if [ -f "$NOHUP_FILE" ]; then
        NOHUP_PROGRESS=$(tail -20 "$NOHUP_FILE" 2>/dev/null | grep "Generating incidents:" | tail -1)
        if [ -n "$NOHUP_PROGRESS" ]; then
            PROGRESS_PAIR=$(echo "$NOHUP_PROGRESS" | grep -o '[0-9]\+/[0-9]\+' | tail -1)
            if [ -n "$PROGRESS_PAIR" ]; then
                CURRENT_EVENTS=$(echo "$PROGRESS_PAIR" | cut -d'/' -f1)
                TOTAL_EVENTS=$(echo "$PROGRESS_PAIR" | cut -d'/' -f2)
                if [ -n "$CURRENT_EVENTS" ] && [ -n "$TOTAL_EVENTS" ] && [ "$TOTAL_EVENTS" -gt 0 ]; then
                    PROGRESS=$(echo "scale=1; $CURRENT_EVENTS * 100 / $TOTAL_EVENTS" | bc 2>/dev/null || echo "0")
                    echo -e "${BOLD}${GREEN}${ROCKET} STATUS: Generation active (${PROGRESS}% complete)${NC}"
                    REALTIME_STATUS="true"
                fi
            fi
        fi
    fi
    
    # Fall back to checkpoint if no real-time progress
    if [ -z "$REALTIME_STATUS" ] && [ -f "$CHECKPOINT_FILE" ]; then
        STATUS=$(cat "$CHECKPOINT_FILE" | jq -r '.status // "running"' 2>/dev/null || echo "running")
        if [ "$STATUS" = "completed" ]; then
            echo -e "${BOLD}${YELLOW}${REFRESH} STATUS: New generation starting (resuming from previous checkpoint)${NC}"
        else
            COMPLETED=$(cat "$CHECKPOINT_FILE" | jq -r '.last_completed_event // 0' 2>/dev/null || echo "0")
            TOTAL=$(cat "$CHECKPOINT_FILE" | jq -r '.total_events // 0' 2>/dev/null || echo "0")
            if [ "$TOTAL" -gt 0 ]; then
                PROGRESS=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc 2>/dev/null || echo "0")
                echo -e "${BOLD}${GREEN}${ROCKET} STATUS: Generation in progress (${PROGRESS}% complete)${NC}"
            else
                echo -e "${BOLD}${GREEN}${ROCKET} STATUS: Generation in progress${NC}"
            fi
        fi
    elif [ -z "$REALTIME_STATUS" ]; then
        # No checkpoint file and no real-time progress
        if [ -f "$NOHUP_FILE" ]; then
            LLM_LOADING=$(tail -20 "$NOHUP_FILE" 2>/dev/null | grep -E "(Initializing LLM|Loading model)" | tail -1)
            if [ -n "$LLM_LOADING" ]; then
                echo -e "${BOLD}${CYAN}${ROBOT} STATUS: Loading LLM model...${NC}"
            else
                echo -e "${BOLD}${YELLOW}${ROCKET} STATUS: Generation starting up...${NC}"
            fi
        else
            echo -e "${BOLD}${YELLOW}${ROCKET} STATUS: Generation starting up...${NC}"
        fi
    fi
else
    # Process not running
    if [ -f "$CHECKPOINT_FILE" ] && [ "$(cat "$CHECKPOINT_FILE" | jq -r '.status' 2>/dev/null)" = "completed" ]; then
        echo -e "${BOLD}${GREEN}${CHECKMARK} STATUS: Generation completed successfully${NC}"
    else
        echo -e "${BOLD}${RED}${CROSS} STATUS: No active generation${NC}"
    fi
fi

# Ensure color reset at end of display
echo -e "${NC}"

}

# Main execution logic
if [[ -n "$WATCH_MODE" ]]; then
    # Auto-refresh mode
    echo -e "${BOLD}${GREEN}Starting auto-refresh monitor (${WATCH_INTERVAL}s intervals)${NC}"
    echo -e "${CYAN}Press Ctrl+C to exit${NC}"
    echo
    
    # Set up signal handling for clean exit
    trap 'echo -e "\n${BOLD}${YELLOW}Monitor stopped${NC}"; exit 0' INT TERM
    
    # Main watch loop
    while true; do
        display_monitor
        sleep "$WATCH_INTERVAL"
    done
else
    # Single run mode
    display_monitor
fi