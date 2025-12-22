#!/bin/bash
# monitor.sh - Comprehensive monitoring for dual GPU weak labels agent

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Clear screen
clear

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Weak Labels Agent - System Monitor${NC}"
echo -e "${CYAN}  $(date)${NC}"
echo -e "${CYAN}================================================${NC}\n"

# ============================================================
# GPU Status
# ============================================================
echo -e "${YELLOW}[1/7] GPU Status${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | head -2 | while IFS=, read -r idx name util_gpu util_mem mem_used mem_total temp power; do
        
        # Trim whitespace
        util_gpu=$(echo $util_gpu | xargs)
        util_mem=$(echo $util_mem | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)
        power=$(echo $power | xargs)
        
        # Color based on utilization
        if [ $util_gpu -gt 70 ]; then
            color=$GREEN
        elif [ $util_gpu -gt 30 ]; then
            color=$YELLOW
        else
            color=$RED
        fi
        
        echo -e "  GPU ${idx}: ${name}"
        echo -e "    Utilization: ${color}${util_gpu}%${NC} | Memory: ${util_mem}% (${mem_used}/${mem_total} MB)"
        echo -e "    Temp: ${temp}°C | Power: ${power}W"
        echo ""
    done
else
    echo -e "  ${RED}✗ nvidia-smi not found${NC}"
fi

# ============================================================
# CPU & Memory Status
# ============================================================
echo -e "${YELLOW}[2/7] CPU & Memory Status${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

# CPU usage
cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
cpu_cores=$(nproc)
echo -e "  CPU Usage: ${GREEN}${cpu_usage}%${NC} (${cpu_cores} cores)"

# Memory usage
mem_info=$(free -h | awk '/^Mem:/ {printf "%.1f/%.1f GB (%.0f%%)", $3, $2, ($3/$2)*100}')
echo -e "  Memory: ${GREEN}${mem_info}${NC}"

# Load average
load_avg=$(uptime | awk -F'load average:' '{print $2}')
echo -e "  Load Average:${load_avg}"
echo ""

# ============================================================
# LLM Servers Health
# ============================================================
echo -e "${YELLOW}[3/7] LLM Servers Health${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

# Check GPU 0 server
if curl -s --max-time 2 http://127.0.0.1:8080/health | grep -q "ok"; then
    echo -e "  GPU 0 (port 8080): ${GREEN}✓ Healthy${NC}"
    
    # Get metrics if available
    slots=$(curl -s --max-time 2 http://127.0.0.1:8080/metrics 2>/dev/null | grep -o '"slots_idle":[0-9]*' | grep -o '[0-9]*')
    if [ ! -z "$slots" ]; then
        echo -e "    Idle slots: $slots/64"
    fi
else
    echo -e "  GPU 0 (port 8080): ${RED}✗ Down or unreachable${NC}"
fi

# Check GPU 1 server
if curl -s --max-time 2 http://127.0.0.1:8081/health | grep -q "ok"; then
    echo -e "  GPU 1 (port 8081): ${GREEN}✓ Healthy${NC}"
    
    # Get metrics if available
    slots=$(curl -s --max-time 2 http://127.0.0.1:8081/metrics 2>/dev/null | grep -o '"slots_idle":[0-9]*' | grep -o '[0-9]*')
    if [ ! -z "$slots" ]; then
        echo -e "    Idle slots: $slots/64"
    fi
else
    echo -e "  GPU 1 (port 8081): ${RED}✗ Down or unreachable${NC}"
fi

# Check server processes
gpu0_pid=$(pgrep -f "llama-server.*8080" | head -1)
gpu1_pid=$(pgrep -f "llama-server.*8081" | head -1)

if [ ! -z "$gpu0_pid" ]; then
    echo -e "  GPU 0 Process: ${GREEN}Running (PID: $gpu0_pid)${NC}"
else
    echo -e "  GPU 0 Process: ${RED}Not running${NC}"
fi

if [ ! -z "$gpu1_pid" ]; then
    echo -e "  GPU 1 Process: ${GREEN}Running (PID: $gpu1_pid)${NC}"
else
    echo -e "  GPU 1 Process: ${RED}Not running${NC}"
fi

echo ""

# ============================================================
# Agent Status
# ============================================================
echo -e "${YELLOW}[4/7] Agent Status${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

if pgrep -f "weak_labels.cli" > /dev/null; then
    agent_pid=$(pgrep -f "weak_labels.cli")
    echo -e "  Agent: ${GREEN}✓ Running (PID: $agent_pid)${NC}"
    
    # Get latest progress
    if [ -f "logs/agent.log" ]; then
        latest_progress=$(grep -E "Progress:|queries processed|ETA" logs/agent.log | tail -1)
        if [ ! -z "$latest_progress" ]; then
            echo -e "  Latest: $latest_progress"
        fi
    fi
else
    echo -e "  Agent: ${RED}✗ Not running${NC}"
fi

echo ""

# ============================================================
# Recent Errors & Warnings (last 5 minutes)
# ============================================================
echo -e "${YELLOW}[5/7] Recent Errors & Warnings (last 5 min)${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

if [ -f "logs/agent.log" ]; then
    # Get errors from last 5 minutes
    errors=$(grep -E "ERROR|CRITICAL" logs/agent.log | tail -5)
    warnings=$(grep -E "WARNING.*timeout|WARNING.*ReadError|WARNING.*failed" logs/agent.log | tail -10)
    
    error_count=$(echo "$errors" | grep -c "ERROR" || echo "0")
    warning_count=$(echo "$warnings" | grep -c "WARNING" || echo "0")
    
    if [ "$error_count" -gt 0 ]; then
        echo -e "  ${RED}✗ Errors: $error_count${NC}"
        echo "$errors" | tail -3 | sed 's/^/    /'
    else
        echo -e "  ${GREEN}✓ No errors${NC}"
    fi
    
    if [ "$warning_count" -gt 0 ]; then
        echo -e "  ${YELLOW}⚠ Warnings: $warning_count${NC}"
        
        # Count timeouts
        timeout_count=$(echo "$warnings" | grep -c "timeout" || echo "0")
        readerror_count=$(echo "$warnings" | grep -c "ReadError" || echo "0")
        
        echo -e "    Timeouts: $timeout_count | ReadErrors: $readerror_count"
        
        # Show last warning
        last_warning=$(echo "$warnings" | tail -1 | cut -d'|' -f4-)
        echo -e "    Last:$last_warning"
    else
        echo -e "  ${GREEN}✓ No warnings${NC}"
    fi
else
    echo -e "  ${RED}✗ Agent log not found${NC}"
fi

echo ""

# ============================================================
# Performance Stats
# ============================================================
echo -e "${YELLOW}[6/7] Performance Stats${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

if [ -f "logs/agent.log" ]; then
    # Cache hit rate (last 100 lines)
    cache_hits=$(tail -100 logs/agent.log | grep -o "Cache: [0-9]* hit" | grep -o "[0-9]*" | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')
    cache_misses=$(tail -100 logs/agent.log | grep -o "[0-9]* miss" | grep -o "[0-9]*" | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')
    
    if [ "$cache_hits" -gt 0 ] || [ "$cache_misses" -gt 0 ]; then
        total=$((cache_hits + cache_misses))
        hit_rate=$((cache_hits * 100 / total))
        echo -e "  Cache Hit Rate: ${GREEN}${hit_rate}%${NC} (avg ${cache_hits} hits, ${cache_misses} misses per batch)"
    fi
    
    # Queries processed
    total_queries=$(grep -o "queries processed" logs/agent.log | wc -l)
    if [ "$total_queries" -gt 0 ]; then
        echo -e "  Total Queries Processed: ${GREEN}${total_queries}+${NC}"
    fi
    
    # Throughput (if available)
    throughput=$(grep -o "[0-9.]* queries/min" logs/agent.log | tail -1)
    if [ ! -z "$throughput" ]; then
        echo -e "  Current Throughput: ${GREEN}${throughput}${NC}"
    fi
    
    # LLM server response time (from server logs)
    if [ -f "llama_server_gpu0.log" ]; then
        avg_time=$(grep "total time" llama_server_gpu0.log | tail -20 | grep -o "total time = *[0-9.]*" | grep -o "[0-9.]*" | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count}')
        if [ ! -z "$avg_time" ]; then
            echo -e "  Avg LLM Response: ${GREEN}${avg_time}ms${NC}"
        fi
    fi
else
    echo -e "  ${RED}✗ No performance data available${NC}"
fi

echo ""

# ============================================================
# Output Files
# ============================================================
echo -e "${YELLOW}[7/7] Output Files${NC}"
echo -e "${BLUE}────────────────────────────────────────────────${NC}"

if [ -f "data/output/qrels.tsv" ]; then
    qrels_lines=$(wc -l < data/output/qrels.tsv)
    qrels_size=$(du -h data/output/qrels.tsv | cut -f1)
    echo -e "  qrels.tsv: ${GREEN}${qrels_lines} lines (${qrels_size})${NC}"
else
    echo -e "  qrels.tsv: ${YELLOW}Not created yet${NC}"
fi

if [ -f "data/output/triples.jsonl" ]; then
    triples_lines=$(wc -l < data/output/triples.jsonl)
    triples_size=$(du -h data/output/triples.jsonl | cut -f1)
    echo -e "  triples.jsonl: ${GREEN}${triples_lines} lines (${triples_size})${NC}"
else
    echo -e "  triples.jsonl: ${YELLOW}Not created yet${NC}"
fi

# Cache
if [ -f "data/cache/llm_judgments.db" ]; then
    cache_size=$(du -h data/cache/llm_judgments.db | cut -f1)
    cache_entries=$(sqlite3 data/cache/llm_judgments.db "SELECT COUNT(*) FROM judgments" 2>/dev/null || echo "N/A")
    echo -e "  LLM Cache: ${GREEN}${cache_entries} judgments (${cache_size})${NC}"
fi

echo ""

# ============================================================
# Summary & Recommendations
# ============================================================
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}Summary & Status${NC}"
echo -e "${CYAN}================================================${NC}"

# Overall health check
issues=0

# Check GPUs
gpu0_ok=$(curl -s --max-time 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q "ok" && echo "1" || echo "0")
gpu1_ok=$(curl -s --max-time 2 http://127.0.0.1:8081/health 2>/dev/null | grep -q "ok" && echo "1" || echo "0")

if [ "$gpu0_ok" = "0" ] || [ "$gpu1_ok" = "0" ]; then
    echo -e "${RED}⚠ Issue: One or both LLM servers are down${NC}"
    ((issues++))
fi

# Check agent
if ! pgrep -f "weak_labels.cli" > /dev/null; then
    echo -e "${RED}⚠ Issue: Agent is not running${NC}"
    ((issues++))
fi

# Check recent errors
if [ -f "logs/agent.log" ]; then
    recent_errors=$(grep -c "ERROR" logs/agent.log | tail -50 || echo "0")
    if [ "$recent_errors" -gt 5 ]; then
        echo -e "${RED}⚠ Issue: High error rate ($recent_errors recent errors)${NC}"
        ((issues++))
    fi
fi

if [ $issues -eq 0 ]; then
    echo -e "${GREEN}✓ All systems operational!${NC}"
    echo -e "${GREEN}  Everything is running smoothly.${NC}"
else
    echo -e "${YELLOW}⚠ $issues issue(s) detected. Check logs above.${NC}"
fi

echo ""
echo -e "${CYAN}Quick Commands:${NC}"
echo -e "  Agent log:     ${YELLOW}tail -f logs/agent.log${NC}"
echo -e "  GPU 0 log:     ${YELLOW}tail -f llama_server_gpu0.log${NC}"
echo -e "  GPU 1 log:     ${YELLOW}tail -f llama_server_gpu1.log${NC}"
echo -e "  Live GPU:      ${YELLOW}watch -n 1 nvidia-smi${NC}"
echo -e "  Restart:       ${YELLOW}bash run_agent_hpc.sh${NC}"
echo ""
