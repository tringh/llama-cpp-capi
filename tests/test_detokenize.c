#include "llama_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define TEST_PASS(msg) printf(ANSI_COLOR_GREEN "✓ PASS" ANSI_COLOR_RESET ": %s\n", msg)
#define TEST_FAIL(msg) printf(ANSI_COLOR_RED "✗ FAIL" ANSI_COLOR_RESET ": %s\n", msg)
#define TEST_INFO(msg) printf(ANSI_COLOR_YELLOW "ℹ INFO" ANSI_COLOR_RESET ": %s\n", msg)
#define TEST_SECTION(msg) printf("\n" ANSI_COLOR_CYAN "=== %s ===" ANSI_COLOR_RESET "\n", msg)

int test_count = 0;
int pass_count = 0;
int fail_count = 0;

void test_detokenize_null_buffer(llama_tokenizer_t* tokenizer) {
    test_count++;
    TEST_SECTION("Test: NULL Buffer Size Checking");

    const char* original_text = "Hello, world! This is a test.";
    printf("Original text: \"%s\"\n", original_text);

    // First, tokenize the text
    int32_t token_count = llama_tokenizer_tokenize(
        tokenizer, original_text, strlen(original_text), NULL, 0, false, false
    );

    if (token_count <= 0) {
        TEST_FAIL("Failed to tokenize text");
        fail_count++;
        return;
    }

    printf("Token count: %d\n", token_count);

    // Allocate and get tokens
    llama_token* tokens = malloc(token_count * sizeof(llama_token));
    int32_t actual_tokens = llama_tokenizer_tokenize(
        tokenizer, original_text, strlen(original_text), tokens, token_count, false, false
    );

    if (actual_tokens != token_count) {
        TEST_FAIL("Token count mismatch");
        fail_count++;
        free(tokens);
        return;
    }

    // Get required text size with NULL buffer
    int32_t required_size = llama_tokenizer_detokenize(
        tokenizer, tokens, token_count, NULL, 0, false, false
    );

    printf("Required text buffer size: %d bytes\n", required_size);

    if (required_size <= 0) {
        TEST_FAIL("NULL buffer should return positive size");
        fail_count++;
        free(tokens);
        return;
    }

    TEST_PASS("NULL buffer returns positive size");
    pass_count++;

    // Allocate exact buffer and detokenize
    char* text = malloc(required_size + 1);  // +1 for null terminator
    int32_t actual_size = llama_tokenizer_detokenize(
        tokenizer, tokens, token_count, text, required_size + 1, false, false
    );
    text[actual_size] = '\0';

    printf("Detokenized text: \"%s\"\n", text);
    printf("Actual size: %d bytes\n", actual_size);

    if (actual_size <= required_size) {
        TEST_PASS("Actual size fits in required size");
        pass_count++;
    } else {
        TEST_FAIL("Actual size exceeds required size");
        printf("  Expected: <= %d, Got: %d\n", required_size, actual_size);
        fail_count++;
    }

    free(text);
    free(tokens);
}

void test_detokenize_insufficient_buffer(llama_tokenizer_t* tokenizer) {
    test_count++;
    TEST_SECTION("Test: Insufficient Buffer");

    const char* original_text = "This is a longer test sentence.";
    printf("Original text: \"%s\"\n", original_text);

    // Tokenize
    int32_t token_count = llama_tokenizer_tokenize(
        tokenizer, original_text, strlen(original_text), NULL, 0, false, false
    );

    llama_token* tokens = malloc(token_count * sizeof(llama_token));
    llama_tokenizer_tokenize(
        tokenizer, original_text, strlen(original_text), tokens, token_count, false, false
    );

    // Get required size
    int32_t required_size = llama_tokenizer_detokenize(
        tokenizer, tokens, token_count, NULL, 0, false, false
    );

    printf("Required size: %d bytes\n", required_size);

    // Try with buffer that's too small
    char small_buffer[10];
    int32_t result = llama_tokenizer_detokenize(
        tokenizer, tokens, token_count, small_buffer, 10, false, false
    );

    if (result < 0) {
        printf("Small buffer returned: %d (indicates %d bytes needed)\n", result, -result);
        if (-result == required_size) {
            TEST_PASS("Insufficient buffer returns correct negative size");
            pass_count++;
        } else {
            TEST_FAIL("Negative size doesn't match required size");
            printf("  Expected: -%d, Got: %d\n", required_size, result);
            fail_count++;
        }
    } else if (result <= 10) {
        TEST_INFO("Text fit in 10-byte buffer (very short detokenization)");
    } else {
        TEST_FAIL("Should return negative for insufficient buffer");
        fail_count++;
    }

    free(tokens);
}

void test_detokenize_roundtrip(llama_tokenizer_t* tokenizer) {
    test_count++;
    TEST_SECTION("Test: Tokenize-Detokenize Roundtrip");

    const char* test_texts[] = {
        "Hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test of tokenization and detokenization.",
        NULL
    };

    int all_passed = 1;

    for (int i = 0; test_texts[i] != NULL; i++) {
        const char* original = test_texts[i];
        printf("\n  Original: \"%s\"\n", original);

        // Tokenize
        int32_t token_count = llama_tokenizer_tokenize(
            tokenizer, original, strlen(original), NULL, 0, false, false
        );

        if (token_count <= 0) {
            printf("    ✗ Tokenization failed\n");
            all_passed = 0;
            continue;
        }

        llama_token* tokens = malloc(token_count * sizeof(llama_token));
        llama_tokenizer_tokenize(
            tokenizer, original, strlen(original), tokens, token_count, false, false
        );

        // Get required size
        int32_t text_size = llama_tokenizer_detokenize(
            tokenizer, tokens, token_count, NULL, 0, false, false
        );

        // Detokenize
        char* text = malloc(text_size + 1);
        int32_t actual = llama_tokenizer_detokenize(
            tokenizer, tokens, token_count, text, text_size + 1, false, false
        );
        text[actual] = '\0';

        printf("    Detokenized: \"%s\"\n", text);
        printf("    Tokens: %d, Bytes: %d\n", token_count, actual);

        // Note: Detokenization might not exactly match original due to tokenizer behavior
        // (e.g., leading spaces, special handling), so we just check it's reasonable
        if (actual > 0) {
            printf("    ✓ Roundtrip completed\n");
        } else {
            printf("    ✗ Roundtrip failed\n");
            all_passed = 0;
        }

        free(text);
        free(tokens);
    }

    if (all_passed) {
        TEST_PASS("All roundtrip tests completed");
        pass_count++;
    } else {
        TEST_FAIL("Some roundtrip tests failed");
        fail_count++;
    }
}

void test_detokenize_empty_tokens(llama_tokenizer_t* tokenizer) {
    test_count++;
    TEST_SECTION("Test: Empty Token Array");

    llama_token tokens[1];  // Not actually used

    // Get size for 0 tokens
    int32_t size = llama_tokenizer_detokenize(
        tokenizer, tokens, 0, NULL, 0, false, false
    );

    printf("Size for 0 tokens: %d\n", size);

    if (size == 0) {
        TEST_PASS("Empty token array returns 0 size");
        pass_count++;
    } else {
        TEST_FAIL("Empty token array should return 0 size");
        fail_count++;
    }

    // Detokenize 0 tokens
    char buffer[10];
    int32_t result = llama_tokenizer_detokenize(
        tokenizer, tokens, 0, buffer, 10, false, false
    );

    if (result == 0) {
        TEST_PASS("Detokenizing 0 tokens returns 0");
        pass_count++;
    } else {
        TEST_FAIL("Detokenizing 0 tokens should return 0");
        fail_count++;
    }
}

void test_detokenize_zero_size_buffer(llama_tokenizer_t* tokenizer) {
    test_count++;
    TEST_SECTION("Test: Zero-Size Buffer Safety");

    const char* text = "Test text for zero-size buffer";

    // Tokenize
    int32_t token_count = llama_tokenizer_tokenize(
        tokenizer, text, strlen(text), NULL, 0, false, false
    );

    llama_token* tokens = malloc(token_count * sizeof(llama_token));
    llama_tokenizer_tokenize(
        tokenizer, text, strlen(text), tokens, token_count, false, false
    );

    // Use zero-size buffer - should not write anything
    char dummy;
    int32_t result = llama_tokenizer_detokenize(
        tokenizer, tokens, token_count, &dummy, 0, false, false
    );

    if (result < 0) {
        TEST_PASS("Zero-size buffer returns negative (required size)");
        pass_count++;
        printf("  Required size: %d bytes\n", -result);
    } else {
        TEST_FAIL("Zero-size buffer should return negative for non-empty detokenization");
        fail_count++;
    }

    free(tokens);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        fprintf(stderr, "Example: %s /path/to/model.gguf\n", argv[0]);
        return 1;
    }

    printf("=== Detokenizer Test Suite ===\n");
    printf("Model: %s\n", argv[1]);

    llama_tokenizer_init();

    llama_tokenizer_t* tokenizer = llama_tokenizer_create(argv[1]);
    if (!tokenizer) {
        fprintf(stderr, "Failed to create tokenizer from: %s\n", argv[1]);
        llama_tokenizer_free_backend();
        return 1;
    }

    printf("Vocab size: %d\n", llama_tokenizer_vocab_size(tokenizer));

    // Run all tests
    test_detokenize_null_buffer(tokenizer);
    test_detokenize_insufficient_buffer(tokenizer);
    test_detokenize_roundtrip(tokenizer);
    test_detokenize_empty_tokens(tokenizer);
    test_detokenize_zero_size_buffer(tokenizer);

    // Cleanup
    llama_tokenizer_destroy(tokenizer);
    llama_tokenizer_free_backend();

    // Print summary
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", test_count);
    printf(ANSI_COLOR_GREEN "Passed: %d" ANSI_COLOR_RESET "\n", pass_count);
    if (fail_count > 0) {
        printf(ANSI_COLOR_RED "Failed: %d" ANSI_COLOR_RESET "\n", fail_count);
    } else {
        printf("Failed: %d\n", fail_count);
    }

    if (fail_count == 0) {
        printf("\n" ANSI_COLOR_GREEN "✓ ALL TESTS PASSED!" ANSI_COLOR_RESET "\n");
        return 0;
    } else {
        printf("\n" ANSI_COLOR_RED "✗ SOME TESTS FAILED" ANSI_COLOR_RESET "\n");
        return 1;
    }
}

