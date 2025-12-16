#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="$HOME/Documents/GitHub/llama.cpp"
MODELS_DIR="$PROJECT_ROOT/models"
MODEL_NAME="Llama-3.2-3B-Instruct-Q8_0.gguf"
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
LLAMA_SERVER_PORT=8080
LLAMA_SERVER_PID_FILE="$PROJECT_ROOT/llama_server.pid"
PASSAGES_PATH="$PROJECT_ROOT/data/passages/corpus_passages.jsonl"

# ============================================================
# Cleanup function - runs on script exit
# ============================================================
cleanup() {
    EXIT_CODE=$?
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    if [ -f "$LLAMA_SERVER_PID_FILE" ]; then
        PID=$(cat "$LLAMA_SERVER_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping llama-server (PID: $PID)...${NC}"
            kill "$PID" 2>/dev/null || true
            sleep 2
            
            if ps -p "$PID" > /dev/null 2>&1; then
                echo -e "${YELLOW}Force stopping...${NC}"
                kill -9 "$PID" 2>/dev/null || true
            fi
            
            echo -e "${GREEN}✓ llama-server stopped${NC}"
        fi
        rm -f "$LLAMA_SERVER_PID_FILE"
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Script completed successfully${NC}"
    else
        echo -e "${RED}✗ Script exited with error (code: $EXIT_CODE)${NC}"
    fi
}

trap cleanup EXIT INT TERM

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Weak Labels - Automated Agent Runner${NC}"
echo -e "${GREEN}================================================${NC}\n"

# ============================================================
# Step 1: Check CMake
# ============================================================
echo -e "${YELLOW}[1/5] Checking build dependencies...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${YELLOW}⚠ CMake not found. Installing...${NC}"
    brew install cmake
fi
CMAKE_VERSION=$(cmake --version | head -n1)
echo -e "${GREEN}✓ CMake: $CMAKE_VERSION${NC}"

# ============================================================
# Step 2: Check/Build llama.cpp
# ============================================================
echo -e "\n${YELLOW}[2/5] Checking llama.cpp...${NC}"

if [ -d "$LLAMA_CPP_DIR" ]; then
    if [ -f "$LLAMA_CPP_DIR/build/bin/llama-server" ]; then
        LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
        echo -e "${GREEN}✓ llama-server found${NC}"
    else
        echo -e "${YELLOW}⚠ Building llama-server...${NC}"
        cd "$LLAMA_CPP_DIR"
        git pull origin master || true
        rm -rf build && mkdir build && cd build
        cmake .. -DGGML_METAL=ON
        cmake --build . --config Release -j$(sysctl -n hw.ncpu)
        LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
    fi
else
    echo -e "${RED}✗ llama.cpp not found at $LLAMA_CPP_DIR${NC}"
    echo "Install: cd ~/Documents/GitHub && git clone https://github.com/ggerganov/llama.cpp"
    exit 1
fi

# ============================================================
# Step 3: Check/Download Model
# ============================================================
echo -e "\n${YELLOW}[3/5] Checking model...${NC}"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo -e "${GREEN}✓ Model found ($MODEL_SIZE)${NC}"
else
    echo -e "${YELLOW}⚠ Model not found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}Clearing models directory and downloading...${NC}"
    
    # Clear all models
    rm -f "$MODELS_DIR"/*.gguf
    
    # Download model
    echo -e "${YELLOW}Downloading $MODEL_NAME from Hugging Face...${NC}"
    echo -e "${YELLOW}This may take 5-10 minutes (~3.3 GB)${NC}\n"
    
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL" --progress=bar:force 2>&1
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL" --progress-bar
    else
        echo -e "${RED}✗ Neither wget nor curl found. Please install one of them.${NC}"
        exit 1
    fi
    
    # Verify download
    if [ -f "$MODEL_PATH" ]; then
        MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        echo -e "\n${GREEN}✓ Model downloaded successfully ($MODEL_SIZE)${NC}"
    else
        echo -e "${RED}✗ Model download failed${NC}"
        exit 1
    fi
fi

# ============================================================
# Step 4: Start llama-server
# ============================================================
echo -e "\n${YELLOW}[4/5] Starting llama-server...${NC}"

if [ -f "$LLAMA_SERVER_PID_FILE" ]; then
    OLD_PID=$(cat "$LLAMA_SERVER_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Already running (PID: $OLD_PID)${NC}"
        SKIP_SERVER=true
    else
        rm -f "$LLAMA_SERVER_PID_FILE"
        SKIP_SERVER=false
    fi
else
    SKIP_SERVER=false
fi

if [ "$SKIP_SERVER" = false ]; then
    lsof -ti:$LLAMA_SERVER_PORT | xargs kill -9 2>/dev/null || true
    
    nohup "$LLAMA_SERVER_BIN" \
        -m "$MODEL_PATH" \
        --ctx-size 1024 \
        --n-predict 10 \
        -ngl 99 \
        --no-mmap \
        --batch-size 256 \
        --ubatch-size 128 \
        --parallel 20 \
        --cont-batching \
        --host 127.0.0.1 \
        --port $LLAMA_SERVER_PORT \
        > "$PROJECT_ROOT/llama_server.log" 2>&1 &
    
    echo $! > "$LLAMA_SERVER_PID_FILE"
    echo -e "${GREEN}✓ Started (PID: $(cat $LLAMA_SERVER_PID_FILE))${NC}"
    
    echo "Waiting for server..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:$LLAMA_SERVER_PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server ready${NC}"
            break
        fi
        [ $i -eq 30 ] && echo -e "${RED}✗ Timeout${NC}" && exit 1
        echo -n "."
        sleep 2
    done
fi

# ============================================================
# Step 5: Setup Python Environment
# ============================================================
echo -e "\n${YELLOW}[5/5] Setting up Python environment...${NC}"

cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    
    if command -v python3.11 &> /dev/null; then
        echo -e "${YELLOW}Using Python 3.11...${NC}"
        python3.11 -m venv .venv
    else
        python3 -m venv .venv
    fi
    
    source .venv/bin/activate
    
    pip install --upgrade pip --quiet
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt --quiet
    
    echo -e "${GREEN}✓ Python environment created${NC}"
else
    source .venv/bin/activate
    echo -e "${GREEN}✓ Using existing virtual environment${NC}"
fi

PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ Virtual environment ready ($PYTHON_VERSION)${NC}"

if [ -f "$PASSAGES_PATH" ]; then
    PASSAGE_COUNT=$(wc -l < "$PASSAGES_PATH" | tr -d ' ')
    echo -e "${GREEN}✓ Passages found ($PASSAGE_COUNT passages)${NC}"
    echo -e "${GREEN}  BM25 index will be built automatically if needed${NC}"
else
    echo -e "${YELLOW}⚠ Passages not found at $PASSAGES_PATH${NC}"
    echo -e "${YELLOW}  Will be created on first agent run from corpus dataset${NC}"
fi

# ============================================================
# Run Agent
# ============================================================
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}  All Systems Ready - Starting Agent${NC}"
echo -e "${GREEN}================================================${NC}\n"

echo -e "Environment:"
echo -e "  Python: ${GREEN}$(python --version | cut -d' ' -f2)${NC}"
echo -e "  LLM: ${GREEN}Llama-3.2-3B (YES/NO mode)${NC}"
echo -e "  LLM Server: ${GREEN}http://127.0.0.1:$LLAMA_SERVER_PORT${NC}"
echo -e "  BM25: ${GREEN}BM25S (pure Python)${NC}\n"

echo -e "Monitor logs:"
echo -e "  LLM: ${YELLOW}tail -f llama_server.log${NC}"
echo -e "  Agent: ${YELLOW}tail -f logs/agent.log${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop (server will auto-cleanup)${NC}\n"

python3 -m weak_labels.cli

echo -e "\n${GREEN}✓ Agent finished successfully${NC}"
