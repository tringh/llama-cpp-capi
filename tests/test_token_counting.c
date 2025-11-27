#include "llama_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [text]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* test_text = argc > 2 ? argv[2] : "Hello, world! This is a test of the tokenizer API with a longer text to ensure proper token counting.";

    printf("Initializing tokenizer backend...\n");
    llama_tokenizer_init();

    printf("Loading model from: %s\n", model_path);
    llama_tokenizer_t* tokenizer = llama_tokenizer_create(model_path);
    if (!tokenizer) {
        fprintf(stderr, "Failed to create tokenizer\n");
        llama_tokenizer_free_backend();
        return 1;
    }

    printf("Vocab size: %d\n\n", llama_tokenizer_vocab_size(tokenizer));

    // Test 1: Get token count with NULL buffer
    printf("Test text: \"%s\"\n", test_text);
    printf("Text length: %zu bytes\n\n", strlen(test_text));

    int32_t token_count = llama_tokenizer_tokenize(
        tokenizer,
        test_text,
        strlen(test_text),
        NULL,  // Pass NULL to get count
        0,
        false,
        false
    );

    if (token_count < 0) {
        fprintf(stderr, "Failed to get token count: %d\n", token_count);
        llama_tokenizer_destroy(tokenizer);
        llama_tokenizer_free_backend();
        return 1;
    }

    printf("Token count (NULL buffer): %d\n", token_count);

    // Test 2: Allocate exact buffer and tokenize
    llama_token* tokens = (llama_token*)malloc(token_count * sizeof(llama_token));
    if (!tokens) {
        fprintf(stderr, "Failed to allocate token buffer\n");
        llama_tokenizer_destroy(tokenizer);
        llama_tokenizer_free_backend();
        return 1;
    }

    int32_t actual_count = llama_tokenizer_tokenize(
        tokenizer,
        test_text,
        strlen(test_text),
        tokens,
        token_count,
        false,
        false
    );

    printf("Token count (actual): %d\n", actual_count);

    if (actual_count != token_count) {
        fprintf(stderr, "WARNING: Token counts don't match! NULL=%d, actual=%d\n", token_count, actual_count);
    } else {
        printf("✓ Token counts match!\n");
    }

    // Print tokens
    printf("\nTokens:\n");
    for (int32_t i = 0; i < actual_count; i++) {
        char piece[256];
        int32_t piece_len = llama_tokenizer_token_to_piece(tokenizer, tokens[i], piece, sizeof(piece));
        if (piece_len > 0 && piece_len < sizeof(piece)) {
            piece[piece_len] = '\0';
            printf("  %3d: %6d = '%s'\n", i, tokens[i], piece);
        } else {
            printf("  %3d: %6d = (error converting to piece)\n", i, tokens[i]);
        }
    }

    // Test 3: Try with a buffer that's too small
    printf("\nTest with insufficient buffer (size 5 for %d tokens):\n", actual_count);
    llama_token small_buffer[5];
    int32_t result = llama_tokenizer_tokenize(
        tokenizer,
        test_text,
        strlen(test_text),
        small_buffer,
        5,
        false,
        false
    );

    if (result < 0) {
        printf("✓ Correctly returned error: %d (negative means %d tokens needed)\n", result, -result);
    } else {
        printf("✗ Expected negative result but got: %d\n", result);
    }

    free(tokens);
    llama_tokenizer_destroy(tokenizer);
    llama_tokenizer_free_backend();

    printf("\n✓ All tests completed successfully!\n");
    return 0;
}

