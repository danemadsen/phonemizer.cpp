#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <cstdio>

#define MAX_TOKENS 256
#define MAX_SEQUENCE_LENGTH 50
#define MAX_TOKEN_LENGTH 32

typedef struct {
    char tokens[MAX_TOKENS][MAX_TOKEN_LENGTH];
    int num_tokens;

    char special_tokens[MAX_TOKENS][MAX_TOKEN_LENGTH];
    int num_special_tokens;

    int char_repeats;
    bool lowercase;
    int pad_index;
    int end_index;
} sequence_tokenizer_params;

void tokenizer_init(
    sequence_tokenizer_params *tokenizer,
    const char **symbols, int num_symbols,
    const char **languages, int num_languages,
    int char_repeats,
    bool lowercase
) {
    strcpy(tokenizer->tokens[0], " ");
    strcpy(tokenizer->special_tokens[0], " ");
    tokenizer->pad_index = 0;
    tokenizer->num_tokens = 1;
    tokenizer->num_special_tokens = 1;
    tokenizer->char_repeats = char_repeats;
    tokenizer->lowercase = lowercase;

    for (int i = 0; i < num_languages; i++) {
        char lang_token[MAX_TOKEN_LENGTH];
        snprintf(lang_token, MAX_TOKEN_LENGTH, "<%s>", languages[i]);

        strcpy(tokenizer->tokens[tokenizer->num_tokens], lang_token);
        strcpy(tokenizer->special_tokens[tokenizer->num_special_tokens], lang_token);
        tokenizer->num_tokens++;
        tokenizer->num_special_tokens++;
    }

    strcpy(tokenizer->tokens[tokenizer->num_tokens], "<end>");
    tokenizer->end_index = tokenizer->num_tokens;
    tokenizer->num_tokens++;

    for (int i = 0; i < num_symbols; i++) {
        strcpy(tokenizer->tokens[tokenizer->num_tokens], symbols[i]);
        tokenizer->num_tokens++;
    }
}

int tokenizer_get_token(const sequence_tokenizer_params *tokenizer, const char *token) {
    for (int i = 0; i < tokenizer->num_tokens; i++) {
        if (strcmp(tokenizer->tokens[i], token) == 0) {
            return i;
        }
    }
    return -1;
}

void tokenizer_encode(
    const sequence_tokenizer_params *tokenizer,
    const char *sentence,
    const char *language,
    int64_t *output
) {
    int idx = 0;
    char processed[1024];
    strncpy(processed, sentence, sizeof(processed));
    processed[sizeof(processed) - 1] = '\0';

    if (tokenizer->lowercase) {
        for (char *p = processed; *p; p++) {
            *p = tolower(*p);
        }
    }

    char symbol[2] = {0};
    for (size_t i = 0; i < strlen(processed); i++) {
        symbol[0] = processed[i];
        int token_index = tokenizer_get_token(tokenizer, symbol);
        if (token_index != -1) {
            for (int j = 0; j < tokenizer->char_repeats && idx < MAX_SEQUENCE_LENGTH; j++) {
                output[idx++] = token_index;
            }
        }
    }

    char lang_token[MAX_TOKEN_LENGTH];
    snprintf(lang_token, MAX_TOKEN_LENGTH, "<%s>", language);
    int lang_index = tokenizer_get_token(tokenizer, lang_token);

    // insert language token at front and end token at back
    memmove(&output[1], &output[0], sizeof(int64_t) * idx);
    output[0] = lang_index;
    idx++;

    if (idx < MAX_SEQUENCE_LENGTH) {
        output[idx++] = tokenizer->end_index;
    }

    while (idx < MAX_SEQUENCE_LENGTH) {
        output[idx++] = tokenizer->pad_index;
    }
}

void tokenizer_decode(const sequence_tokenizer_params *tokenizer, const int64_t *sequence, int len, char output[][MAX_TOKEN_LENGTH], int *out_len) {
    int idx = 0;

    strncpy(output[idx++], tokenizer->tokens[sequence[0]], MAX_TOKEN_LENGTH);

    for (int i = 1; i < len - 1; i += tokenizer->char_repeats) {
        if (sequence[i] == tokenizer->end_index) break;
        strncpy(output[idx++], tokenizer->tokens[sequence[i]], MAX_TOKEN_LENGTH);
    }

    *out_len = idx;
}

int tokenizer_clean(const sequence_tokenizer_params *tokenizer, const int64_t *input, int len, int64_t *output) {
    int out_idx = 0;
    for (int i = 0; i < len; i++) {
        int is_special = 0;
        for (int j = 0; j < tokenizer->num_special_tokens; j++) {
            if (input[i] == tokenizer_get_token(tokenizer, tokenizer->special_tokens[j])) {
                is_special = 1;
                break;
            }
        }
        if (!is_special) {
            output[out_idx++] = input[i];
        }
    }

    // truncate at end_index
    for (int i = 0; i < out_idx; i++) {
        if (output[i] == tokenizer->end_index) {
            out_idx = i;
            break;
        }
    }

    // remove consecutive duplicates
    int cleaned_idx = 1;
    for (int i = 1; i < out_idx; i++) {
        if (output[i] != output[i - 1]) {
            output[cleaned_idx++] = output[i];
        }
    }

    return cleaned_idx;
}

#endif // TOKENIZER_H
