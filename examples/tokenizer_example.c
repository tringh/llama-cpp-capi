#include "llama_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char* program) {
    printf("Usage: %s <model.gguf> [text]\n", program);
    printf("\n");
    printf("Example:\n");
    printf("  %s model.gguf \"Hello, world!\"\n", program);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* text = argc > 2 ? argv[2] : "Hello, world!";

    printf("===========================================\n");
    printf("llama-cpp-capi Tokenizer Example\n");
    printf("===========================================\n");
    printf("Model: %s\n", model_path);
    printf("Text:  \"%s\"\n", text);
    printf("===========================================\n\n");

    // Initialize tokenizer backend
    llama_tokenizer_init();

    // Create tokenizer
    printf("Loading tokenizer...\n");
    llama_tokenizer_t* tokenizer = llama_tokenizer_create(model_path);
    if (!tokenizer) {
        fprintf(stderr, "Error: Failed to create tokenizer\n");
        llama_tokenizer_free_backend();
        return 1;
    }

    printf("Tokenizer loaded successfully!\n\n");

    // Get vocabulary info
    int32_t vocab_size = llama_tokenizer_vocab_size(tokenizer);
    llama_token bos = llama_tokenizer_token_bos(tokenizer);
    llama_token eos = llama_tokenizer_token_eos(tokenizer);
    llama_token nl = llama_tokenizer_token_nl(tokenizer);

    printf("Vocabulary Info:\n");
    printf("  Size:       %d\n", vocab_size);
    printf("  BOS token:  %d\n", bos);
    printf("  EOS token:  %d\n", eos);
    printf("  NL token:   %d\n", nl);
    printf("\n");

    // Tokenize
    int text_len = strlen(text);

    // First call to get token count
    int n_tokens = llama_tokenizer_tokenize(
        tokenizer, text, text_len,
        NULL, 0, true, false
    );

    if (n_tokens < 0) {
        fprintf(stderr, "Error: Tokenization failed\n");
        llama_tokenizer_destroy(tokenizer);
        llama_tokenizer_free_backend();
        return 1;
    }

    printf("Tokenization:\n");
    printf("  Input length:    %d characters\n", text_len);
    printf("  Number of tokens: %d\n\n", n_tokens);

    // Allocate and tokenize
    llama_token* tokens = (llama_token*)malloc(n_tokens * sizeof(llama_token));
    if (!tokens) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        llama_tokenizer_destroy(tokenizer);
        llama_tokenizer_free_backend();
        return 1;
    }

    llama_tokenizer_tokenize(
        tokenizer, text, text_len,
        tokens, n_tokens, true, false
    );

    // Print tokens
    printf("Tokens:\n");
    char piece_buf[256];
    for (int i = 0; i < n_tokens; i++) {
        int piece_len = llama_tokenizer_token_to_piece(
            tokenizer, tokens[i],
            piece_buf, sizeof(piece_buf) - 1
        );

        if (piece_len > 0) {
            piece_buf[piece_len] = '\0';
            // Replace special chars for display
            printf("  [%2d] ID: %6d  Text: '", i, tokens[i]);
            for (int j = 0; j < piece_len; j++) {
                char c = piece_buf[j];
                if (c == ' ') printf("·");
                else if (c == '\n') printf("↵");
                else if (c == '\t') printf("→");
                else printf("%c", c);
            }
            printf("'\n");
        } else {
            printf("  [%2d] ID: %6d  Text: <error>\n", i, tokens[i]);
        }
    }

    // Detokenize
    printf("\nDetokenization:\n");
    char output[4096];
    int output_len = llama_tokenizer_detokenize(
        tokenizer, tokens, n_tokens,
        output, sizeof(output) - 1
    );

    if (output_len >= 0) {
        output[output_len] = '\0';
        printf("  Output: \"%s\"\n", output);
        printf("  Length: %d bytes\n", output_len);

        if (strcmp(text, output) == 0) {
            printf("  Match:  ✓ Perfect match!\n");
        } else {
            printf("  Match:  ✗ Different (may be due to normalization)\n");
        }
    } else {
        printf("  Error: Detokenization failed\n");
    }

    // Cleanup
    free(tokens);
    llama_tokenizer_destroy(tokenizer);
    llama_tokenizer_free_backend();

    printf("\n===========================================\n");
    printf("Example completed successfully!\n");
    printf("===========================================\n");

    return 0;
}

