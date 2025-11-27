#include "llama_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    llama_tokenizer_init();
    llama_tokenizer_t* tokenizer = llama_tokenizer_create(argv[1]);
    if (!tokenizer) {
        fprintf(stderr, "Failed to create tokenizer\n");
        llama_tokenizer_free_backend();
        return 1;
    }

    // Test cases with different text lengths
    const char* test_cases[] = {
        "",                                    // Empty
        "a",                                   // Single char
        "Hello",                               // Short word
        "Hello, world!",                       // Sentence
        "This is a longer test sentence.",    // Longer text
        "The quick brown fox jumps over the lazy dog. This is a much longer sentence with many more tokens to test the buffer overflow behavior properly.",
        NULL
    };

    for (int i = 0; test_cases[i] != NULL; i++) {
        const char* text = test_cases[i];
        int32_t text_len = strlen(text);

        printf("\n=== Test Case %d ===\n", i);
        printf("Text: \"%s\"\n", text);
        printf("Length: %d bytes\n", text_len);

        // Method 1: Get count with NULL buffer
        int32_t count_from_null = llama_tokenizer_tokenize(
            tokenizer, text, text_len, NULL, 0, false, false
        );
        printf("Count from NULL buffer: %d\n", count_from_null);

        // Method 2: Get count with actual tokenization
        if (count_from_null > 0) {
            llama_token* tokens = malloc(count_from_null * sizeof(llama_token));
            if (tokens) {
                int32_t actual_count = llama_tokenizer_tokenize(
                    tokenizer, text, text_len, tokens, count_from_null, false, false
                );
                printf("Actual token count: %d\n", actual_count);

                if (actual_count != count_from_null) {
                    printf("❌ MISMATCH: NULL gave %d, actual is %d\n", count_from_null, actual_count);
                } else {
                    printf("✅ Counts match\n");
                }

                free(tokens);
            }
        }

        // Method 3: Test with insufficient buffer
        llama_token small_buf[2];
        int32_t result_small = llama_tokenizer_tokenize(
            tokenizer, text, text_len, small_buf, 2, false, false
        );
        printf("Result with 2-token buffer: %d\n", result_small);

        if (result_small < 0) {
            printf("  (indicates %d tokens needed)\n", -result_small);
        }
    }

    llama_tokenizer_destroy(tokenizer);
    llama_tokenizer_free_backend();
    return 0;
}

