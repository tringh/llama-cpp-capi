#ifndef LLAMA_TOKENIZER_H
#define LLAMA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to a tokenizer instance
 */
typedef struct llama_tokenizer_t llama_tokenizer_t;

/**
 * Token type (32-bit integer)
 */
typedef int32_t llama_token;

/**
 * Initialize the tokenizer backend
 * Must be called before any other tokenizer functions
 */
void llama_tokenizer_init(void);

/**
 * Free the tokenizer backend
 * Should be called when done using tokenizer functions
 */
void llama_tokenizer_free_backend(void);

/**
 * Create a tokenizer from a GGUF model file
 *
 * @param model_path Path to the GGUF model file
 * @return Tokenizer handle, or NULL on failure
 */
llama_tokenizer_t* llama_tokenizer_create(const char* model_path);

/**
 * Free a tokenizer instance
 *
 * @param tokenizer Tokenizer handle to free
 */
void llama_tokenizer_destroy(llama_tokenizer_t* tokenizer);

/**
 * Get the vocabulary size
 *
 * @param tokenizer Tokenizer handle
 * @return Number of tokens in vocabulary
 */
int32_t llama_tokenizer_vocab_size(const llama_tokenizer_t* tokenizer);

/**
 * Get special tokens
 *
 * @param tokenizer Tokenizer handle
 * @return Special token ID, or -1 if not available
 */
llama_token llama_tokenizer_token_bos(const llama_tokenizer_t* tokenizer);  // beginning-of-sentence
llama_token llama_tokenizer_token_eos(const llama_tokenizer_t* tokenizer);  // end-of-sentence
llama_token llama_tokenizer_token_eot(const llama_tokenizer_t* tokenizer);  // end-of-turn
llama_token llama_tokenizer_token_nl(const llama_tokenizer_t* tokenizer);   // newline
llama_token llama_tokenizer_token_pad(const llama_tokenizer_t* tokenizer);  // padding
llama_token llama_tokenizer_token_sep(const llama_tokenizer_t* tokenizer);  // separator

/**
 * Get FIM (Fill-In-Middle) special tokens for code completion
 *
 * @param tokenizer Tokenizer handle
 * @return FIM token ID, or -1 if not available
 */
llama_token llama_tokenizer_token_fim_pre(const llama_tokenizer_t* tokenizer);  // prefix
llama_token llama_tokenizer_token_fim_suf(const llama_tokenizer_t* tokenizer);  // suffix
llama_token llama_tokenizer_token_fim_mid(const llama_tokenizer_t* tokenizer);  // middle

/**
 * Check token properties
 *
 * @param tokenizer Tokenizer handle
 * @param token Token to check
 * @return true if token has the property, false otherwise
 */
bool llama_tokenizer_is_eog(const llama_tokenizer_t* tokenizer, llama_token token);      // end-of-generation
bool llama_tokenizer_is_control(const llama_tokenizer_t* tokenizer, llama_token token);  // control token

/**
 * Get token metadata
 *
 * @param tokenizer Tokenizer handle
 * @param token Token to query
 * @return Token text/score, or NULL/-1 on error
 */
const char* llama_tokenizer_token_get_text(const llama_tokenizer_t* tokenizer, llama_token token);
float llama_tokenizer_token_get_score(const llama_tokenizer_t* tokenizer, llama_token token);

/**
 * Check if special tokens should be added automatically
 *
 * @param tokenizer Tokenizer handle
 * @return true if special tokens should be added
 */
bool llama_tokenizer_should_add_bos(const llama_tokenizer_t* tokenizer);
bool llama_tokenizer_should_add_eos(const llama_tokenizer_t* tokenizer);

/**
 * Tokenize text into tokens
 *
 * @param tokenizer Tokenizer handle
 * @param text Text to tokenize
 * @param text_len Length of text in bytes
 * @param tokens Output buffer for tokens (can be NULL to get count)
 * @param n_max_tokens Maximum number of tokens to write
 * @param add_special Whether to add special tokens
 * @param parse_special Whether to parse special tokens in text
 * @return Number of tokens, or negative on error
 */
int32_t llama_tokenizer_tokenize(
    const llama_tokenizer_t* tokenizer,
    const char* text,
    int32_t text_len,
    llama_token* tokens,
    int32_t n_max_tokens,
    bool add_special,
    bool parse_special
);

/**
 * Convert a single token to text
 *
 * @param tokenizer Tokenizer handle
 * @param token Token to convert
 * @param buf Output buffer for text
 * @param length Size of output buffer
 * @return Number of bytes written, or negative on error
 */
int32_t llama_tokenizer_token_to_piece(
    const llama_tokenizer_t* tokenizer,
    llama_token token,
    char* buf,
    int32_t length
);

/**
 * Detokenize tokens to text
 *
 * @param tokenizer Tokenizer handle
 * @param tokens Array of tokens
 * @param n_tokens Number of tokens
 * @param text Output buffer for text
 * @param text_len Size of output buffer
 * @return Number of bytes written, or negative on error
 */
int32_t llama_tokenizer_detokenize(
    const llama_tokenizer_t* tokenizer,
    const llama_token* tokens,
    int32_t n_tokens,
    char* text,
    int32_t text_len
);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TOKENIZER_H

