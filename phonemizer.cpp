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

static std::string format(const char *fmt, ...) {
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

static int get_key_idx(const gguf_context *ctx, const char *key) {
    int i = gguf_find_key(ctx, key);
    if (i == -1) {
        printf("key %s not found in file\n", key);
    }
    return i;
}

static int32_t get_i32(const gguf_context *ctx, const std::string &key) {
    const int i = get_key_idx(ctx, key.c_str());
    return gguf_get_val_i32(ctx, i);
}

static float get_f32(const gguf_context *ctx, const std::string &key) {
    const int i = get_key_idx(ctx, key.c_str());
    return gguf_get_val_f32(ctx, i);
}

static struct ggml_tensor *get_embedding(struct ggml_context *ctx, struct ggml_tensor *weight, struct ggml_tensor *input) {
    int n_embed = weight->ne[0];
    int n_tokens = input->ne[1];
    struct ggml_tensor *output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, n_tokens);
    printf("Embedding: n_embed=%d, n_tokens=%d\n", n_embed, n_tokens);

    int32_t *input_data = (int32_t *)input->data;

    for (int i = 0; i < n_tokens; ++i) {
        int idx = input_data[i];
        printf("Embedding: idx=%d\n", idx);
        if (idx < 0 || idx >= weight->ne[0]) {
            printf("Error: idx=%d out of bounds\n", idx);
            return nullptr;
        }
        memcpy((float *)output->data + i * n_embed, (float *)weight->data + idx * n_embed, n_embed * sizeof(float));
    }
    return output;
}

static struct ggml_tensor *get_positional_encoding(struct ggml_context *ctx, struct ggml_tensor *input, int d_model, float dropout) {
    int n_tokens = input->ne[1];
    struct ggml_tensor *positional_encoding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);
    float *pos_data = (float *)positional_encoding->data;
    for (int pos = 0; pos < n_tokens; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            if (i % 2 == 0) {
                pos_data[pos * d_model + i] = sin(pos / pow(10000, i / (double)d_model));
            } else {
                pos_data[pos * d_model + i] = cos(pos / pow(10000, (i - 1) / (double)d_model));
            }
        }
    }
    struct ggml_tensor *output = ggml_add(ctx, input, positional_encoding);
    return output;
}

static struct ggml_tensor * get_linear(struct ggml_context * ctx, struct ggml_tensor * W, struct ggml_tensor * b, struct ggml_tensor * input) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, W, input);
    out = ggml_add(ctx, out, b);
    return out;
}

static struct ggml_tensor * get_transformer_encoder_layer(struct ggml_context * ctx, struct ggml_tensor * input, int n_heads, int d_model, int d_ff) {
    int d_head = d_model / n_heads;

    // Input shape
    printf("Input shape: [%lld, %lld]\n", input->ne[0], input->ne[1]);

    // Self-attention weights
    struct ggml_tensor * Wq = ggml_get_tensor(ctx, "Wq");
    struct ggml_tensor * bq = ggml_get_tensor(ctx, "bq");
    struct ggml_tensor * Wk = ggml_get_tensor(ctx, "Wk");
    struct ggml_tensor * bk = ggml_get_tensor(ctx, "bk");
    struct ggml_tensor * Wv = ggml_get_tensor(ctx, "Wv");
    struct ggml_tensor * bv = ggml_get_tensor(ctx, "bv");

    // Initialize tensors if they are nullptr
    int64_t d_model_dims[2] = {d_model, d_model};
    int64_t d_model_single_dim[1] = {d_model};
    int64_t d_ff_dims[2] = {d_ff, d_model};
    int64_t d_ff_single_dim[1] = {d_ff};

    if (!Wq) Wq = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, d_model_dims);
    if (!bq) bq = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, d_model_single_dim);
    if (!Wk) Wk = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, d_model_dims);
    if (!bk) bk = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, d_model_single_dim);
    if (!Wv) Wv = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, d_model_dims);
    if (!bv) bv = ggml_new_tensor(ctx, GGML_TYPE_F32, 1, d_model_single_dim);

    // Self-attention
    struct ggml_tensor * Q = get_linear(ctx, Wq, bq, input); // Query
    struct ggml_tensor * K = get_linear(ctx, Wk, bk, input); // Key
    struct ggml_tensor * V = get_linear(ctx, Wv, bv, input); // Value

    printf("Q shape: [%lld, %lld]\n", Q->ne[0], Q->ne[1]);
    printf("K shape: [%lld, %lld]\n", K->ne[0], K->ne[1]);
    printf("V shape: [%lld, %lld]\n", V->ne[0], V->ne[1]);

    // Reshape tensors to [batch_size, n_heads, seq_len, d_head]
    struct ggml_tensor * Q_reshaped = ggml_reshape_4d(ctx, Q, d_head, n_heads, Q->ne[1], Q->ne[0] / d_model);
    struct ggml_tensor * K_reshaped = ggml_reshape_4d(ctx, K, d_head, n_heads, K->ne[1], K->ne[0] / d_model);
    struct ggml_tensor * V_reshaped = ggml_reshape_4d(ctx, V, d_head, n_heads, V->ne[1], V->ne[0] / d_model);

    printf("Q_reshaped shape: [%lld, %lld, %lld, %lld]\n", Q_reshaped->ne[0], Q_reshaped->ne[1], Q_reshaped->ne[2], Q_reshaped->ne[3]);
    printf("K_reshaped shape: [%lld, %lld, %lld, %lld]\n", K_reshaped->ne[0], K_reshaped->ne[1], K_reshaped->ne[2], K_reshaped->ne[3]);
    printf("V_reshaped shape: [%lld, %lld, %lld, %lld]\n", V_reshaped->ne[0], V_reshaped->ne[1], V_reshaped->ne[2], V_reshaped->ne[3]);

    GGML_ASSERT(Q_reshaped->ne[0] * Q_reshaped->ne[1] * Q_reshaped->ne[2] * Q_reshaped->ne[3] == Q->ne[0] * Q->ne[1]);
    GGML_ASSERT(K_reshaped->ne[0] * K_reshaped->ne[1] * K_reshaped->ne[2] * K_reshaped->ne[3] == K->ne[0] * K->ne[1]);
    GGML_ASSERT(V_reshaped->ne[0] * V_reshaped->ne[1] * V_reshaped->ne[2] * V_reshaped->ne[3] == V->ne[0] * V->ne[1]);

    struct ggml_tensor * QK = ggml_mul_mat(ctx, Q_reshaped, K_reshaped);

    printf("QK shape: [%lld, %lld, %lld, %lld]\n", QK->ne[0], QK->ne[1], QK->ne[2], QK->ne[3]);
    
    struct ggml_tensor * attn_weights = ggml_soft_max(ctx, ggml_scale(ctx, QK, 1.0 / sqrt(d_head)));

    printf("attn_weights shape: [%lld, %lld, %lld, %lld]\n", attn_weights->ne[0], attn_weights->ne[1], attn_weights->ne[2], attn_weights->ne[3]);

    struct ggml_tensor * attn_output = ggml_mul_mat(ctx, attn_weights, V_reshaped);

    // Reshape attention output back to [batch_size, seq_len, d_model]
    struct ggml_tensor * attn_output_reshaped = ggml_reshape_2d(ctx, attn_output, d_model, attn_output->ne[2]);

    struct ggml_tensor * W_o = ggml_get_tensor(ctx, "W_o");
    struct ggml_tensor * b_o = ggml_get_tensor(ctx, "b_o");
    struct ggml_tensor * attn_output_proj = get_linear(ctx, W_o, b_o, attn_output_reshaped);

    // Add & Norm
    struct ggml_tensor * attn_output_norm = ggml_norm(ctx, ggml_add(ctx, input, attn_output_proj), 1e-6);

    // Feed Forward weights
    struct ggml_tensor * W_ff1 = ggml_get_tensor(ctx, "W_ff1");
    struct ggml_tensor * b_ff1 = ggml_get_tensor(ctx, "b_ff1");
    struct ggml_tensor * W_ff2 = ggml_get_tensor(ctx, "W_ff2");
    struct ggml_tensor * b_ff2 = ggml_get_tensor(ctx, "b_ff2");

    // Feed Forward
    struct ggml_tensor * ff_hidden = ggml_relu(ctx, get_linear(ctx, W_ff1, b_ff1, attn_output_norm));
    struct ggml_tensor * ff_output = get_linear(ctx, W_ff2, b_ff2, ff_hidden);

    // Add & Norm
    struct ggml_tensor * encoder_output = ggml_norm(ctx, ggml_add(ctx, attn_output_norm, ff_output), 1e-6);

    return encoder_output;
}

struct ggml_cgraph *create_graph(struct ggml_context *ctx, const phonemizer_model &model, struct ggml_tensor *input_tensor) {
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    printf("Creating graph...\n");

    struct ggml_tensor *x = input_tensor;

    // Ensure embedding.weight tensor exists
    if (model.tensors.find("embedding.weight") == model.tensors.end()) {
        printf("Error: embedding.weight tensor not found\n");
        return nullptr;
    }

    x = get_embedding(ctx, model.tensors.at("embedding.weight"), x);
    printf("Embedding done\n");

    x = get_positional_encoding(ctx, x, model.hparams.d_model, model.hparams.dropout);
    printf("Positional encoding done\n");

    for (int i = 0; i < model.hparams.layers; ++i) {
        std::string layer_prefix = "encoder.layers." + std::to_string(i);
        printf("Layer prefix: %s\n", layer_prefix.c_str());
        x = get_transformer_encoder_layer(
            ctx,
            x,
            model.hparams.heads,
            model.hparams.d_model,
            model.hparams.d_fft
        );
    }
    printf("Transformer encoder done\n");

    x = get_linear(ctx, model.tensors.at("fc_out.weight"), model.tensors.at("fc_out.bias"), x);
    printf("Linear layer done\n");

    ggml_set_name(x, "result");
    printf("Result tensor named\n");

    ggml_build_forward_expand(gf, x);
    printf("Forward graph expanded\n");

    return gf;
}

struct ggml_tensor *compute(const phonemizer_model &model, const std::vector<float> &input) {
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

    struct ggml_cgraph *gf = create_graph(ctx, model, input_tensor);
    if (!gf) {
        printf("Error during graph creation\n");
        ggml_free(ctx);
        return nullptr;
    }
    printf("GRAPH CREATION DONE\n");

    ggml_graph_compute_with_ctx(ctx, gf, 1);
    printf("GRAPH COMPUTE DONE\n");

    struct ggml_tensor *result = gf->nodes[gf->n_nodes - 1];

    ggml_free(ctx);
    return result;
}

void load_model(const std::string &fname, phonemizer_model &model) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    struct ggml_context *meta = NULL;

    struct gguf_init_params gguf_params = {
        true,
        &meta,
    };

    struct gguf_context *ctx = gguf_init_from_file(fname.c_str(), gguf_params);

    if (!ctx) {
        throw std::runtime_error(format("%s: failed to open '%s'\n", __func__, fname.c_str()));
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n", __func__, n_kv, n_tensors, fname.c_str());

    size_t model_size = 0;

    for (int i = 0; i < n_tensors; ++i) {
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
    if (!model.backend) {
        throw std::runtime_error(format("%s: ggml_backend_cpu_init() failed\n", __func__));
    }

    struct ggml_init_params ggml_params = {
        model_size,
        NULL,
        true
    };

    model.ctx = ggml_init(ggml_params);

    if (!model.ctx) {
        throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
    }

    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *cur = ggml_dup_tensor(model.ctx, ggml_get_tensor(meta, name));
        ggml_set_name(cur, name);
        model.tensors[name] = cur;
    }

    auto fin = std::ifstream(fname, std::ios::binary);

    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer) {
        throw std::runtime_error(format("%s: failed to allocate memory for the model\n", __func__));
    }

    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(model.ctx, name);
        const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
        fin.seekg(offset, std::ios::beg);
        if (!fin) {
            gguf_free(ctx);
            throw std::runtime_error(format("%s: failed to seek for tensor %s\n", __func__, name));
        }
        int num_bytes = ggml_nbytes(cur);
        if (ggml_backend_buffer_is_host(model.buffer)) {
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
    
    for (auto &entry : model.tensors) {
        const std::string &name = entry.first;
        entry.second = ggml_get_tensor(model.ctx, name.c_str());
    }

    gguf_free(ctx);
}
