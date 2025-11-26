# llama-cpp-capi

Clean C API wrapper for llama.cpp tokenization.

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/llama-cpp-capi.git

# Build the library
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build examples (optional, independent)
cmake -B examples/build examples
cmake --build examples/build

# Run example
./examples/build/tokenizer_example path/to/model.gguf "Hello, world!"
```

## Requirements

- CMake 3.14+
- C++17 compiler
- C11 compiler (for examples)
