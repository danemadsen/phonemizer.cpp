#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <limits.h>
#include <inttypes.h>

static std::string format(const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static int get_key_idx(const gguf_context *ctx, const char *key)
{
    int i = gguf_find_key(ctx, key);
    if (i == -1)
    {
        printf("key %s not found in file\n", key);
    }

    return i;
}

static int32_t get_i32(const gguf_context *ctx, const std::string &key)
{
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_i32(ctx, i);
}

static float get_f32(const gguf_context *ctx, const std::string &key)
{
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_f32(ctx, i);
}

static struct ggml_tensor *get_embedding(struct ggml_context *ctx, struct ggml_tensor *weight, struct ggml_tensor *input)
{
    int n_embed = weight->ne[1];
    int n_tokens = input->ne[0];
    struct ggml_tensor *output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, n_tokens);
    for (int i = 0; i < n_tokens; ++i)
    {
        int idx = ((int32_t *)input->data)[i];
        memcpy((float *)output->data + i * n_embed, (float *)weight->data + idx * n_embed, n_embed * sizeof(float));
    }
    return output;
}

static struct ggml_tensor *get_positional_encoding(struct ggml_context *ctx, struct ggml_tensor *input, int d_model, float dropout)
{
    // Assume max length is the input length
    int n_tokens = input->ne[1];
    struct ggml_tensor *positional_encoding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);

    // Fill positional_encoding with the positional encodings
    float *pos_data = (float *)positional_encoding->data;
    for (int pos = 0; pos < n_tokens; ++pos)
    {
        for (int i = 0; i < d_model; ++i)
        {
            if (i % 2 == 0)
            {
                pos_data[pos * d_model + i] = sin(pos / pow(10000, i / (double)d_model));
            }
            else
            {
                pos_data[pos * d_model + i] = cos(pos / pow(10000, (i - 1) / (double)d_model));
            }
        }
    }

    struct ggml_tensor *output = ggml_add(ctx, input, positional_encoding);
    return output;
}

static struct ggml_tensor *get_linear(struct ggml_context *ctx, struct ggml_tensor *weight, struct ggml_tensor *bias, struct ggml_tensor *input)
{
    struct ggml_tensor *output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input->ne[0], weight->ne[0]);
    struct ggml_tensor *temp = ggml_mul_mat(ctx, weight, input);
    output = ggml_add(ctx, temp, bias);
    return output;
}

// Simplified Transformer Encoder Layer
static struct ggml_tensor *ggml_transformer_encoder_layer(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_tensor *self_attn_q_weight, struct ggml_tensor *self_attn_k_weight, 
                                                          struct ggml_tensor *self_attn_v_weight, struct ggml_tensor *self_attn_out_proj_weight, struct ggml_tensor *linear1_weight, 
                                                          struct ggml_tensor *linear1_bias, struct ggml_tensor *linear2_weight, struct ggml_tensor *linear2_bias, 
                                                          int d_model)
{
    // Implement a simplified transformer encoder layer here.
    // This function should apply the attention mechanism and the feed-forward network with normalization.
    
    // Prepare the query, key, value tensors
    struct ggml_tensor *q = ggml_mul_mat(ctx, input, self_attn_q_weight);
    struct ggml_tensor *k = ggml_mul_mat(ctx, input, self_attn_k_weight);
    struct ggml_tensor *v = ggml_mul_mat(ctx, input, self_attn_v_weight);
    
    // Apply the attention mechanism
    struct ggml_tensor *attention_output = ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f / sqrt(d_model), 0.0f);
    
    // Apply the output projection weight
    attention_output = ggml_mul_mat(ctx, attention_output, self_attn_out_proj_weight);
    
    // Apply normalization to the attention output
    struct ggml_tensor *norm1_output = ggml_norm(ctx, attention_output, 1e-5);
    
    // Feed-forward network
    struct ggml_tensor *feed_forward_intermediate = ggml_mul_mat(ctx, norm1_output, linear1_weight);
    feed_forward_intermediate = ggml_add_inplace(ctx, feed_forward_intermediate, linear1_bias);
    feed_forward_intermediate = ggml_relu_inplace(ctx, feed_forward_intermediate);
    
    struct ggml_tensor *feed_forward_output = ggml_mul_mat(ctx, feed_forward_intermediate, linear2_weight);
    feed_forward_output = ggml_add_inplace(ctx, feed_forward_output, linear2_bias);
    
    // Apply normalization to the feed-forward output
    struct ggml_tensor *output = ggml_norm(ctx, feed_forward_output, 1e-5);
    
    return output;
}