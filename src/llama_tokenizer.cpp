#include "llama_tokenizer.h"
#include "llama.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct llama_tokenizer_t {
    llama_model* model;
    const llama_vocab* vocab;
};

void llama_tokenizer_init(void) {
    llama_backend_init();
}

void llama_tokenizer_free_backend(void) {
    llama_backend_free();
}

llama_tokenizer_t* llama_tokenizer_create(const char* model_path) {
    if (!model_path) {
        return NULL;
    }
    llama_tokenizer_t* tokenizer = (llama_tokenizer_t*)malloc(sizeof(llama_tokenizer_t));
    if (!tokenizer) {
        return NULL;
    }
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    tokenizer->model = llama_model_load_from_file(model_path, params);
    if (!tokenizer->model) {
        free(tokenizer);
        return NULL;
    }
    tokenizer->vocab = llama_model_get_vocab(tokenizer->model);
    if (!tokenizer->vocab) {
        llama_model_free(tokenizer->model);
        free(tokenizer);
        return NULL;
    }
    return tokenizer;
}

void llama_tokenizer_destroy(llama_tokenizer_t* tokenizer) {
    if (tokenizer) {
        if (tokenizer->model) {
            llama_model_free(tokenizer->model);
        }
        free(tokenizer);
    }
}

int32_t llama_tokenizer_vocab_size(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_n_tokens(tokenizer->vocab);
}

llama_token llama_tokenizer_token_bos(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_bos(tokenizer->vocab);
}

llama_token llama_tokenizer_token_eos(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_eos(tokenizer->vocab);
}

llama_token llama_tokenizer_token_eot(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_eot(tokenizer->vocab);
}

llama_token llama_tokenizer_token_nl(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_nl(tokenizer->vocab);
}

llama_token llama_tokenizer_token_pad(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_pad(tokenizer->vocab);
}

llama_token llama_tokenizer_token_sep(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_sep(tokenizer->vocab);
}

llama_token llama_tokenizer_token_fim_pre(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_fim_pre(tokenizer->vocab);
}

llama_token llama_tokenizer_token_fim_suf(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_fim_suf(tokenizer->vocab);
}

llama_token llama_tokenizer_token_fim_mid(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }
    return llama_vocab_fim_mid(tokenizer->vocab);
}

bool llama_tokenizer_is_eog(const llama_tokenizer_t* tokenizer, llama_token token) {
    if (!tokenizer || !tokenizer->vocab) {
        return false;
    }
    return llama_vocab_is_eog(tokenizer->vocab, token);
}

bool llama_tokenizer_is_control(const llama_tokenizer_t* tokenizer, llama_token token) {
    if (!tokenizer || !tokenizer->vocab) {
        return false;
    }
    return llama_vocab_is_control(tokenizer->vocab, token);
}

const char* llama_tokenizer_token_get_text(const llama_tokenizer_t* tokenizer, llama_token token) {
    if (!tokenizer || !tokenizer->vocab) {
        return NULL;
    }
    return llama_vocab_get_text(tokenizer->vocab, token);
}

float llama_tokenizer_token_get_score(const llama_tokenizer_t* tokenizer, llama_token token) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1.0f;
    }
    return llama_vocab_get_score(tokenizer->vocab, token);
}

bool llama_tokenizer_should_add_bos(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return false;
    }
    return llama_vocab_get_add_bos(tokenizer->vocab);
}

bool llama_tokenizer_should_add_eos(const llama_tokenizer_t* tokenizer) {
    if (!tokenizer || !tokenizer->vocab) {
        return false;
    }
    return llama_vocab_get_add_eos(tokenizer->vocab);
}

int32_t llama_tokenizer_tokenize(
    const llama_tokenizer_t* tokenizer,
    const char* text,
    int32_t text_len,
    llama_token* tokens,
    int32_t n_max_tokens,
    bool add_special,
    bool parse_special
) {
    if (!tokenizer || !tokenizer->vocab || !text) {
        return -1;
    }

    // If tokens is NULL, caller wants to know the token count
    // Allocate a temporary buffer to get the count
    if (tokens == NULL) {
        // Reasonable upper bound: each byte could be a token, plus special tokens
        int32_t max_tokens = text_len + 10;
        llama_token* temp_tokens = (llama_token*)malloc(max_tokens * sizeof(llama_token));
        if (!temp_tokens) {
            return -1;
        }

        int32_t result = llama_tokenize(
            tokenizer->vocab,
            text,
            text_len,
            temp_tokens,
            max_tokens,
            add_special,
            parse_special
        );

        free(temp_tokens);
        return result;
    }

    // Normal tokenization with provided buffer
    return llama_tokenize(
        tokenizer->vocab,
        text,
        text_len,
        tokens,
        n_max_tokens,
        add_special,
        parse_special
    );
}

int32_t llama_tokenizer_token_to_piece(
    const llama_tokenizer_t* tokenizer,
    llama_token token,
    char* buf,
    int32_t length
) {
    if (!tokenizer || !tokenizer->vocab) {
        return -1;
    }

    return llama_token_to_piece(
        tokenizer->vocab,
        token,
        buf,
        length,
        0,
        false
    );
}

int32_t llama_tokenizer_detokenize(
    const llama_tokenizer_t* tokenizer,
    const llama_token* tokens,
    int32_t n_tokens,
    char* text,
    int32_t text_len
) {
    if (!tokenizer || !tokenizer->vocab || !tokens || !text) {
        return -1;
    }

    int32_t pos = 0;
    for (int32_t i = 0; i < n_tokens && pos < text_len; i++) {
        int32_t n = llama_token_to_piece(
            tokenizer->vocab,
            tokens[i],
            text + pos,
            text_len - pos,
            0,
            false
        );
        if (n < 0) {
            return -1;
        }
        pos += n;
    }

    if (pos < text_len) {
        text[pos] = '\0';
    }

    return pos;
}
