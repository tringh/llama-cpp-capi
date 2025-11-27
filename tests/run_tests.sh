#!/bin/bash

# Test runner script for llama_tokenizer tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Llama Tokenizer Test Runner ==="
echo ""

# Check if tests are built
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Building tests...${NC}"
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug
    cmake --build "$BUILD_DIR"
    echo ""
fi

# Check if model path is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Model path required${NC}"
    echo "Usage: $0 <path_to_model.gguf>"
    echo ""
    echo "Example:"
    echo "  $0 ~/models/llama-2-7b.Q4_K_M.gguf"
    echo ""
    echo "You can download models from HuggingFace:"
    echo "  https://huggingface.co/models?library=gguf"
    exit 1
fi

MODEL_PATH="$1"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Using model: $MODEL_PATH${NC}"
echo ""

# Run tests
FAILED=0

echo "=========================================="
echo "Running: Edge Cases Test"
echo "=========================================="
if "$BUILD_DIR/test_edge_cases" "$MODEL_PATH"; then
    echo -e "${GREEN}✓ Edge cases test passed${NC}"
else
    echo -e "${RED}✗ Edge cases test failed${NC}"
    FAILED=1
fi
echo ""

echo "=========================================="
echo "Running: Token Counting Test"
echo "=========================================="
if "$BUILD_DIR/test_token_counting" "$MODEL_PATH"; then
    echo -e "${GREEN}✓ Token counting test passed${NC}"
else
    echo -e "${RED}✗ Token counting test failed${NC}"
    FAILED=1
fi
echo ""

echo "=========================================="
echo "Running: Buffer Behavior Test"
echo "=========================================="
if "$BUILD_DIR/test_buffer_behavior" "$MODEL_PATH"; then
    echo -e "${GREEN}✓ Buffer behavior test passed${NC}"
else
    echo -e "${RED}✗ Buffer behavior test failed${NC}"
    FAILED=1
fi
echo ""

echo "=========================================="
echo "Running: Detokenize Test"
echo "=========================================="
if "$BUILD_DIR/test_detokenize" "$MODEL_PATH"; then
    echo -e "${GREEN}✓ Detokenize test passed${NC}"
else
    echo -e "${RED}✗ Detokenize test failed${NC}"
    FAILED=1
fi
echo ""

# Summary
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi

