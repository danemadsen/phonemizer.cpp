#include "ggml.h"
#include "ggml-backend.h"
#include "phonemizer.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <limits.h>
#include <inttypes.h>
#include <cstdarg>
#include <cmath>

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
    printf("Embedding: n_embed=%d, n_tokens=%d\n", n_embed, n_tokens);

    int32_t *input_data = (int32_t *)input->data;

    for (int i = 0; i < n_tokens; ++i)
    {
        int idx = input_data[i];
        printf("Embedding: idx=%d\n", idx);
        if (idx < 0 || idx >= weight->ne[0])
        {
            printf("Error: idx=%d out of bounds\n", idx);
            return nullptr;
        }
        memcpy((float *)output->data + i * n_embed, (float *)weight->data + idx * n_embed, n_embed * sizeof(float));
    }
    return output;
}

static struct ggml_tensor *get_positional_encoding(struct ggml_context *ctx, struct ggml_tensor *input, int d_model, float dropout)
{
    int n_tokens = input->ne[1];
    struct ggml_tensor *positional_encoding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);
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

static struct ggml_tensor *get_transformer_encoder_layer(struct ggml_context *ctx, struct ggml_tensor *input, struct ggml_tensor *self_attn_q_weight, struct ggml_tensor *self_attn_k_weight, 
                                                          struct ggml_tensor *self_attn_v_weight, struct ggml_tensor *self_attn_out_proj_weight, struct ggml_tensor *linear1_weight, 
                                                          struct ggml_tensor *linear1_bias, struct ggml_tensor *linear2_weight, struct ggml_tensor *linear2_bias, 
                                                          int d_model)
{
    struct ggml_tensor *q = ggml_mul_mat(ctx, input, self_attn_q_weight);
    struct ggml_tensor *k = ggml_mul_mat(ctx, input, self_attn_k_weight);
    struct ggml_tensor *v = ggml_mul_mat(ctx, input, self_attn_v_weight);
    
    struct ggml_tensor *attention_output = ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f / sqrt(d_model), 0.0f);
    attention_output = ggml_mul_mat(ctx, attention_output, self_attn_out_proj_weight);
    
    struct ggml_tensor *norm1_output = ggml_norm(ctx, attention_output, 1e-5);
    
    struct ggml_tensor *feed_forward_intermediate = ggml_mul_mat(ctx, norm1_output, linear1_weight);
    feed_forward_intermediate = ggml_add_inplace(ctx, feed_forward_intermediate, linear1_bias);
    feed_forward_intermediate = ggml_relu_inplace(ctx, feed_forward_intermediate);
    
    struct ggml_tensor *feed_forward_output = ggml_mul_mat(ctx, feed_forward_intermediate, linear2_weight);
    feed_forward_output = ggml_add_inplace(ctx, feed_forward_output, linear2_bias);
    
    struct ggml_tensor *output = ggml_norm(ctx, feed_forward_output, 1e-5);
    
    return output;
}

struct ggml_tensor *forward(struct ggml_tensor *input_tensor, struct ggml_context *ctx, const phonemizer_model &model)
{
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    printf("FORWARD START\n");

    struct ggml_tensor *x = input_tensor;

    // Ensure embedding.weight tensor exists
    if (model.tensors.find("embedding.weight") == model.tensors.end())
    {
        printf("Error: embedding.weight tensor not found\n");
        return nullptr;
    }

    x = get_embedding(ctx, model.tensors.at("embedding.weight"), x);
    if (!x) return nullptr; // Check for null
    printf("EMBEDDING DONE\n");
    
    x = get_positional_encoding(ctx, x, model.hparams.d_model, model.hparams.dropout);
    printf("POSITIONAL ENCODING DONE\n");

    for (int i = 0; i < model.hparams.layers; ++i)
    {
        std::string layer_prefix = "encoder.layers." + std::to_string(i);
        printf("LAYER PREFIX: %s\n", layer_prefix.c_str());
        x = get_transformer_encoder_layer(
            ctx,
            x,
            model.tensors.at(layer_prefix + ".self_attn.in_proj_weight"),
            model.tensors.at(layer_prefix + ".self_attn.in_proj_bias"),
            model.tensors.at(layer_prefix + ".self_attn.out_proj.weight"),
            model.tensors.at(layer_prefix + ".self_attn.out_proj.bias"),
            model.tensors.at(layer_prefix + ".linear1.weight"),
            model.tensors.at(layer_prefix + ".linear1.bias"),
            model.tensors.at(layer_prefix + ".linear2.weight"),
            model.tensors.at(layer_prefix + ".linear2.bias"),
            model.hparams.d_model
        );
    }
    printf("TRANSFORMER ENCODER DONE\n");

    x = get_linear(ctx, model.tensors.at("fc_out.weight"), model.tensors.at("fc_out.bias"), x);
    printf("LINEAR DONE\n");

    ggml_set_name(x, "result");
    printf("SET NAME DONE\n");

    ggml_build_forward_expand(gf, x);
    printf("BUILD FORWARD EXPAND DONE\n");

    ggml_graph_compute_with_ctx(ctx, gf, 1);
    printf("GRAPH COMPUTE DONE\n");

    return x;
}

struct ggml_tensor *compute(const phonemizer_model &model, const std::vector<float> &input)
{
    int32_t vocab_size = model.hparams.encoder_vocab_size; 

    static size_t buf_size = vocab_size * sizeof(float) * 1024 * 1024;
    static void *buf = malloc(buf_size);
    struct ggml_init_params params = {
        buf_size,
        buf,
        false,
    };
    struct ggml_context *ctx = ggml_init(params);
    
    struct ggml_tensor *input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ((int32_t *)input_tensor->data)[i] = static_cast<int32_t>(input[i]);
    }
    ggml_set_name(input_tensor, "input_tensor");

    printf("MEMCPY DONE\n");

    struct ggml_tensor *result = forward(input_tensor, ctx, model);
    if (!result) {
        printf("Error during forward pass\n");
        ggml_free(ctx);
        return nullptr;
    }
    printf("FORWARD DONE\n");

    ggml_free(ctx);
    return result;
}

void load_model(const std::string &fname, phonemizer_model &model)
{
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    struct ggml_context *meta = NULL;

    struct gguf_init_params gguf_params = {
        true,
        &meta,
    };

    struct gguf_context *ctx = gguf_init_from_file(fname.c_str(), gguf_params);

    if (!ctx)
    {
        throw std::runtime_error(format("%s: failed to open '%s'\n", __func__, fname.c_str()));
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n", __func__, n_kv, n_tensors, fname.c_str());

    size_t model_size = 0;

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        const size_t offset = gguf_get_tensor_offset(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(meta, name);
        size_t tensor_size = ggml_nbytes(cur);
        model_size += tensor_size;

        printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
               __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
    }

    model_size += 10 * 1024 * 1024;

    model.backend = ggml_backend_cpu_init();
    if (!model.backend)
    {
        throw std::runtime_error(format("%s: ggml_backend_cpu_init() failed\n", __func__));
    }

    struct ggml_init_params ggml_params = {
        model_size,
        NULL,
        true
    };

    model.ctx = ggml_init(ggml_params);

    if (!model.ctx)
    {
        throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
    }

    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *cur = ggml_dup_tensor(model.ctx, ggml_get_tensor(meta, name));
        ggml_set_name(cur, name);
        model.tensors[name] = cur;
    }

    auto fin = std::ifstream(fname, std::ios::binary);

    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer)
    {
        throw std::runtime_error(format("%s: failed to allocate memory for the model\n", __func__));
    }

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(model.ctx, name);
        const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
        fin.seekg(offset, std::ios::beg);
        if (!fin)
        {
            gguf_free(ctx);
            throw std::runtime_error(format("%s: failed to seek for tensor %s\n", __func__, name));
        }
        int num_bytes = ggml_nbytes(cur);
        if (ggml_backend_buffer_is_host(model.buffer))
        {
            fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
        }
    }
    fin.close();

    model.hparams.encoder_vocab_size = get_i32(ctx, "encoder_vocab_size");
    model.hparams.decoder_vocab_size = get_i32(ctx, "decoder_vocab_size");
    model.hparams.d_model = get_i32(ctx, "d_model");
    model.hparams.d_fft = get_i32(ctx, "d_fft");
    model.hparams.layers = get_i32(ctx, "layers");
    model.hparams.dropout = get_f32(ctx, "dropout");
    model.hparams.heads = get_i32(ctx, "heads");
    
    for (auto &entry : model.tensors)
    {
        const std::string &name = entry.first;
        entry.second = ggml_get_tensor(model.ctx, name.c_str());
    }

    gguf_free(ctx);
}

