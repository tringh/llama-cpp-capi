#include "llama_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define TEST_PASS(msg) printf(ANSI_COLOR_GREEN "✓ PASS" ANSI_COLOR_RESET ": %s\n", msg)
#define TEST_FAIL(msg) printf(ANSI_COLOR_RED "✗ FAIL" ANSI_COLOR_RESET ": %s\n", msg)
#define TEST_INFO(msg) printf(ANSI_COLOR_YELLOW "ℹ INFO" ANSI_COLOR_RESET ": %s\n", msg)

int test_count = 0;
int pass_count = 0;
int fail_count = 0;

void test_empty_text(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: Empty Text ---\n");

    const char* text = "";
    int32_t text_len = 0;

    // Get count with NULL buffer
    int32_t count = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, false, false);
    printf("Empty text token count: %d\n", count);

    if (count == 0) {
        TEST_PASS("Empty text returns 0 tokens");
        pass_count++;
    } else {
        TEST_FAIL("Empty text should return 0 tokens");
        fail_count++;
        return;
    }

    // Verify actual tokenization
    llama_token tokens[1];
    int32_t actual = llama_tokenizer_tokenize(tokenizer, text, text_len, tokens, 1, false, false);

    if (actual == 0) {
        TEST_PASS("Empty text actual tokenization returns 0");
        pass_count++;
    } else {
        TEST_FAIL("Empty text actual tokenization should return 0");
        fail_count++;
    }
}

void test_single_char(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: Single Character ---\n");

    const char* text = "a";
    int32_t text_len = 1;

    // Get count with NULL buffer
    int32_t count = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, false, false);
    printf("Single char 'a' token count: %d\n", count);

    if (count >= 1) {
        TEST_PASS("Single char returns positive token count");
        pass_count++;
    } else {
        TEST_FAIL("Single char should return positive token count");
        fail_count++;
        return;
    }

    // Verify actual tokenization matches
    llama_token* tokens = malloc(count * sizeof(llama_token));
    int32_t actual = llama_tokenizer_tokenize(tokenizer, text, text_len, tokens, count, false, false);

    if (actual == count) {
        TEST_PASS("Single char: NULL count matches actual tokenization");
        pass_count++;
    } else {
        TEST_FAIL("Single char: count mismatch");
        printf("  Expected: %d, Got: %d\n", count, actual);
        fail_count++;
    }

    free(tokens);
}

void test_short_text(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: Short Text ---\n");

    const char* text = "Hello";
    int32_t text_len = strlen(text);

    // Get count with NULL buffer
    int32_t count = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, false, false);
    printf("Text '%s' token count: %d\n", text, count);

    if (count > 0) {
        TEST_PASS("Short text returns positive token count");
        pass_count++;
    } else {
        TEST_FAIL("Short text should return positive token count");
        fail_count++;
        return;
    }

    // Verify with actual tokenization
    llama_token* tokens = malloc(count * sizeof(llama_token));
    int32_t actual = llama_tokenizer_tokenize(tokenizer, text, text_len, tokens, count, false, false);

    if (actual == count) {
        TEST_PASS("Short text: NULL count matches actual tokenization");
        pass_count++;
    } else {
        TEST_FAIL("Short text: count mismatch");
        printf("  Expected: %d, Got: %d\n", count, actual);
        fail_count++;
    }

    free(tokens);
}

void test_insufficient_buffer(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: Insufficient Buffer ---\n");

    const char* text = "This is a longer sentence with many tokens.";
    int32_t text_len = strlen(text);

    // Get actual count
    int32_t count = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, false, false);
    printf("Text token count: %d\n", count);

    if (count <= 0) {
        TEST_INFO("Text produced 0 tokens, skipping insufficient buffer test");
        return;
    }

    // Try with buffer that's too small
    llama_token small_buffer[3];
    int32_t result = llama_tokenizer_tokenize(tokenizer, text, text_len, small_buffer, 3, false, false);

    if (result < 0 && -result == count) {
        TEST_PASS("Insufficient buffer returns negative of required count");
        pass_count++;
    } else if (result >= 0 && result <= 3) {
        TEST_INFO("Text fit in 3-token buffer (very short tokenization)");
    } else {
        TEST_FAIL("Insufficient buffer behavior incorrect");
        printf("  Expected: -%d, Got: %d\n", count, result);
        fail_count++;
    }
}

void test_null_buffer_consistency(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: NULL Buffer Consistency ---\n");

    const char* texts[] = {
        "",
        "a",
        "Hi",
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test sentence with many different words and tokens to verify consistency.",
        NULL
    };

    int all_consistent = 1;

    for (int i = 0; texts[i] != NULL; i++) {
        const char* text = texts[i];
        int32_t text_len = strlen(text);

        // Get count with NULL
        int32_t count_null = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, false, false);

        if (count_null < 0) {
            printf("  ERROR: NULL buffer returned negative for text[%d]\n", i);
            all_consistent = 0;
            continue;
        }

        // Get count with actual tokenization
        if (count_null > 0) {
            llama_token* tokens = malloc(count_null * sizeof(llama_token));
            int32_t count_actual = llama_tokenizer_tokenize(tokenizer, text, text_len, tokens, count_null, false, false);

            if (count_null != count_actual) {
                printf("  MISMATCH: text[%d] NULL=%d actual=%d\n", i, count_null, count_actual);
                all_consistent = 0;
            }

            free(tokens);
        }
    }

    if (all_consistent) {
        TEST_PASS("NULL buffer count is consistent with actual tokenization for all test cases");
        pass_count++;
    } else {
        TEST_FAIL("NULL buffer count inconsistent with actual tokenization");
        fail_count++;
    }
}

void test_zero_size_buffer_safety(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: Zero Size Buffer Safety ---\n");

    const char* text = "This text should not cause any writes when buffer size is 0.";
    int32_t text_len = strlen(text);

    // Use a dummy buffer with size 0 - should not write anything
    llama_token dummy;
    int32_t result = llama_tokenizer_tokenize(tokenizer, text, text_len, &dummy, 0, false, false);

    if (result < 0) {
        TEST_PASS("Zero size buffer returns negative count (no writes)");
        pass_count++;
        printf("  Required tokens: %d\n", -result);
    } else if (result == 0) {
        TEST_INFO("Text produced 0 tokens");
    } else {
        TEST_FAIL("Zero size buffer should return negative for non-empty tokenization");
        fail_count++;
    }
}

void test_with_special_tokens(llama_tokenizer_t* tokenizer) {
    test_count++;
    printf("\n--- Test: Special Tokens ---\n");

    const char* text = "Hello world";
    int32_t text_len = strlen(text);

    // Without special tokens
    int32_t count_no_special = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, false, false);

    // With special tokens
    int32_t count_with_special = llama_tokenizer_tokenize(tokenizer, text, text_len, NULL, 0, true, false);

    printf("  Without special: %d tokens\n", count_no_special);
    printf("  With special: %d tokens\n", count_with_special);

    if (count_with_special >= count_no_special) {
        TEST_PASS("Special tokens add to or equal token count");
        pass_count++;
    } else {
        TEST_FAIL("Special tokens should not reduce token count");
        fail_count++;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        fprintf(stderr, "Example: %s /path/to/model.gguf\n", argv[0]);
        return 1;
    }

    printf("=== Tokenizer Edge Cases Test Suite ===\n");
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
    test_empty_text(tokenizer);
    test_single_char(tokenizer);
    test_short_text(tokenizer);
    test_insufficient_buffer(tokenizer);
    test_null_buffer_consistency(tokenizer);
    test_zero_size_buffer_safety(tokenizer);
    test_with_special_tokens(tokenizer);

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

