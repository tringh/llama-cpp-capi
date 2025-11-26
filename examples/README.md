# llama-cpp-capi Examples

This directory contains example programs demonstrating how to use the llama-cpp-capi tokenizer library.

## Building Examples

The examples are built **independently** from the main library. You must build the main library first.

### Prerequisites

1. Build the main library:
   ```bash
   cd ..
   cmake -B build
   cmake --build build
   ```

2. The library will be built at: `../build/libllama_tokenizer.so`

### Build Examples

#### Standard build (using default library location)

```bash
cd examples
cmake -B build
cmake --build build
```

The example will be built at: `examples/build/tokenizer_example`

#### Custom Library Location

If you installed the library to a custom location, you can specify the paths:

```bash
cd examples
cmake -B build \
  -DLLAMA_TOKENIZER_LIB_DIR=/path/to/lib \
  -DLLAMA_TOKENIZER_INCLUDE_DIR=/path/to/include
cmake --build build
```

## Running Examples

### tokenizer_example

Basic tokenization example:

```bash
./build/tokenizer_example <model.gguf> [text]
```

Example:
```bash
./build/tokenizer_example ~/models/llama-3-8b.gguf "Hello, world!"
```

## Troubleshooting

### Library Not Found

If you get errors about the library not being found:

1. Ensure the main library is built:
   ```bash
   ls ../build/libllama_tokenizer.so
   ```

2. Specify the correct path when configuring:
   ```bash
   cmake -B build -DLLAMA_TOKENIZER_LIB_DIR=/path/to/lib
   ```

### Header Not Found

If you get errors about missing headers:

1. Ensure the header exists:
   ```bash
   ls ../include/llama_tokenizer.h
   ```

2. Specify the correct path when configuring:
   ```bash
   cmake -B build -DLLAMA_TOKENIZER_INCLUDE_DIR=/path/to/include
   ```

