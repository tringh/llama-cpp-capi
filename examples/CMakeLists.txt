cmake_minimum_required(VERSION 3.14)
project(llama-tokenizer-examples VERSION 0.0.1 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "===========================================")
message(STATUS "llama-cpp-capi Examples")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===========================================")

# Option to specify the library location
set(LLAMA_TOKENIZER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build" CACHE PATH "Path to llama_tokenizer library directory")
set(LLAMA_TOKENIZER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include" CACHE PATH "Path to llama_tokenizer include directory")

message(STATUS "Library directory: ${LLAMA_TOKENIZER_LIB_DIR}")
message(STATUS "Include directory: ${LLAMA_TOKENIZER_INCLUDE_DIR}")

# Find the library
find_library(LLAMA_TOKENIZER_LIBRARY
    NAMES llama_tokenizer
    PATHS ${LLAMA_TOKENIZER_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT LLAMA_TOKENIZER_LIBRARY)
    message(FATAL_ERROR "llama_tokenizer library not found in ${LLAMA_TOKENIZER_LIB_DIR}\n"
                        "Please build the main library first or set LLAMA_TOKENIZER_LIB_DIR")
endif()

message(STATUS "Found library: ${LLAMA_TOKENIZER_LIBRARY}")

# Verify include directory exists
if(NOT EXISTS "${LLAMA_TOKENIZER_INCLUDE_DIR}/llama_tokenizer.h")
    message(FATAL_ERROR "llama_tokenizer.h not found in ${LLAMA_TOKENIZER_INCLUDE_DIR}")
endif()

# Tokenizer example
add_executable(tokenizer_example
    tokenizer_example.c
)

target_include_directories(tokenizer_example
    PRIVATE
        ${LLAMA_TOKENIZER_INCLUDE_DIR}
)

target_link_libraries(tokenizer_example
    PRIVATE
        ${LLAMA_TOKENIZER_LIBRARY}
)

# Set RPATH so the executable can find the shared library
set_target_properties(tokenizer_example PROPERTIES
    BUILD_RPATH "${LLAMA_TOKENIZER_LIB_DIR}"
    INSTALL_RPATH "${LLAMA_TOKENIZER_LIB_DIR}"
)

message(STATUS "===========================================")
message(STATUS "Examples configured:")
message(STATUS "  - tokenizer_example")
message(STATUS "===========================================")

