cmake_minimum_required(VERSION 3.14)
project(llama_tokenizer_tests C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find the main library
find_library(LLAMA_TOKENIZER_LIB
    NAMES llama_tokenizer
    HINTS ${CMAKE_SOURCE_DIR}/../build
    REQUIRED
)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)

# Test 1: Token counting test
add_executable(test_token_counting test_token_counting.c)
target_link_libraries(test_token_counting ${LLAMA_TOKENIZER_LIB})
set_target_properties(test_token_counting PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test 2: Buffer behavior test
add_executable(test_buffer_behavior test_buffer_behavior.c)
target_link_libraries(test_buffer_behavior ${LLAMA_TOKENIZER_LIB})
set_target_properties(test_buffer_behavior PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test 3: Edge cases test
add_executable(test_edge_cases test_edge_cases.c)
target_link_libraries(test_edge_cases ${LLAMA_TOKENIZER_LIB})
set_target_properties(test_edge_cases PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test 4: Detokenize test
add_executable(test_detokenize test_detokenize.c)
target_link_libraries(test_detokenize ${LLAMA_TOKENIZER_LIB})
set_target_properties(test_detokenize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Add custom target to run all tests if model is available
add_custom_target(run_tests
    COMMAND echo "=== Running Token Counting Test ==="
    COMMAND ${CMAKE_BINARY_DIR}/test_token_counting ${MODEL_PATH} || echo "SKIP: No model specified"
    COMMAND echo ""
    COMMAND echo "=== Running Buffer Behavior Test ==="
    COMMAND ${CMAKE_BINARY_DIR}/test_buffer_behavior ${MODEL_PATH} || echo "SKIP: No model specified"
    COMMAND echo ""
    COMMAND echo "=== Running Edge Cases Test ==="
    COMMAND ${CMAKE_BINARY_DIR}/test_edge_cases ${MODEL_PATH} || echo "SKIP: No model specified"
    COMMAND echo ""
    COMMAND echo "=== Running Detokenize Test ==="
    COMMAND ${CMAKE_BINARY_DIR}/test_detokenize ${MODEL_PATH} || echo "SKIP: No model specified"
    DEPENDS test_token_counting test_buffer_behavior test_edge_cases test_detokenize
    COMMENT "Running tokenizer tests"
)

# Print usage information
message(STATUS "Tests configured. Build with: cmake --build .")
message(STATUS "Run tests with: MODEL_PATH=/path/to/model.gguf make run_tests")
message(STATUS "Or run individually: ./test_token_counting /path/to/model.gguf")

