#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
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

struct ggml_cgraph *create_graph(phonemizer_model *model, struct ggml_tensor *input_tokens) {
    struct ggml_context *ctx = model->ctx;
    phonemizer_model_hparams *hp = &model->hparams;

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

    // 1. Embedding: [T, N] → [T, N, D]
    struct ggml_tensor *emb_table = model->tensors["embedding.weight"]; // [V, D]
    struct ggml_tensor *x = ggml_get_rows(ctx, emb_table, input_tokens); // [T, N, D]

    // 2. Positional Encoding
    auto it = model->tensors.find("pos_encoding");
    if (it == model->tensors.end() || it->second == nullptr) {
        fprintf(stderr, "ERROR: pos_encoding tensor missing or null\n");
        exit(1);
    }
    struct ggml_tensor *pos_encoding = it->second; // [T, D]
    struct ggml_tensor *pe_expanded = ggml_repeat(ctx, pos_encoding, x); // [T, N, D]
    x = ggml_add(ctx, x, pe_expanded); // [T, N, D]

    // 3. TransformerEncoder layers
    for (int i = 0; i < hp->layers; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "encoder.layers.%d", i);

        struct ggml_tensor *Wq = model->tensors[std::string(prefix) + ".self_attn.q_proj.weight"];
        struct ggml_tensor *Bq = model->tensors[std::string(prefix) + ".self_attn.q_proj.bias"];
        struct ggml_tensor *Wk = model->tensors[std::string(prefix) + ".self_attn.k_proj.weight"];
        struct ggml_tensor *Bk = model->tensors[std::string(prefix) + ".self_attn.k_proj.bias"];
        struct ggml_tensor *Wv = model->tensors[std::string(prefix) + ".self_attn.v_proj.weight"];
        struct ggml_tensor *Bv = model->tensors[std::string(prefix) + ".self_attn.v_proj.bias"];
        struct ggml_tensor *Wo = model->tensors[std::string(prefix) + ".self_attn.out_proj.weight"];
        struct ggml_tensor *Bo = model->tensors[std::string(prefix) + ".self_attn.out_proj.bias"];

        // Optionally normalize first
        x = ggml_rms_norm(ctx, x, 1e-5f);

        struct ggml_tensor *q = ggml_add(ctx, ggml_mul_mat(ctx, Wq, x), Bq);
        struct ggml_tensor *k = ggml_add(ctx, ggml_mul_mat(ctx, Wk, x), Bk);
        struct ggml_tensor *v = ggml_add(ctx, ggml_mul_mat(ctx, Wv, x), Bv);

        struct ggml_tensor *attn_out = ggml_flash_attn_ext(
            ctx, q, k, v,
            NULL,
            1.0f / sqrtf((float)(hp->d_model / hp->heads)),
            0.0f,
            -INFINITY
        );

        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        attn_out = ggml_add(ctx, ggml_mul_mat(ctx, Wo, attn_out), Bo);
        attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);
        x = ggml_add(ctx, x, attn_out);

        struct ggml_tensor *W1 = model->tensors[std::string(prefix) + ".linear1.weight"];
        struct ggml_tensor *B1 = model->tensors[std::string(prefix) + ".linear1.bias"];
        struct ggml_tensor *W2 = model->tensors[std::string(prefix) + ".linear2.weight"];
        struct ggml_tensor *B2 = model->tensors[std::string(prefix) + ".linear2.bias"];

        struct ggml_tensor *ff = ggml_add(ctx, ggml_mul_mat(ctx, W1, x), B1);
        ff = ggml_relu(ctx, ff);
        ff = ggml_add(ctx, ggml_mul_mat(ctx, W2, ff), B2);

        x = ggml_add(ctx, x, ff);
    }

    // 4. Final projection to decoder_vocab_size
    struct ggml_tensor *fc_w = model->tensors["fc_out.weight"]; // [V_out, D]
    struct ggml_tensor *fc_b = model->tensors["fc_out.bias"];   // [V_out]
    x = ggml_add(ctx, ggml_mul_mat(ctx, fc_w, x), fc_b);        // [T, N, V_out]

    // 5. Transpose back to [N, T, V_out] if needed (not always necessary depending on final usage)
    x = ggml_permute(ctx, x, 1, 0, 2, 3); // [T, N, V] → [N, T, V]

    // Set a known name for the final output tensor
    ggml_set_name(x, "output_tensor");

    // Register final node
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    return gf;
}

struct ggml_tensor *compute(phonemizer_model *model, const std::vector<int64_t> &input) {
    int32_t vocab_size = model->hparams.encoder_vocab_size;

    static size_t buf_size = vocab_size * sizeof(int64_t) * 1024 * 1024;
    static void *buf = malloc(buf_size);
    struct ggml_init_params params = {
        buf_size,
        buf,
        false,
    };
    struct ggml_context *ctx = ggml_init(params);

    struct ggml_tensor *input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ((int32_t *)input_tensor->data)[i] = input[i];
    }
    ggml_set_name(input_tensor, "input_tensor");

    struct ggml_cgraph *gf = create_graph(model, input_tensor);
    if (!gf) {
        fprintf(stderr, "Error during graph creation\n");
        ggml_free(ctx);
        return nullptr;
    }

    if (ggml_backend_graph_compute(model->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute() failed\n");
        ggml_free(ctx);
        return nullptr;
    }

    // Retrieve the computed tensor (last node in the graph)
    struct ggml_tensor *result = ggml_graph_get_tensor(gf, "output_tensor");

    // Copy result tensor to a new context to return safely
    struct ggml_context *result_ctx = ggml_init({
        ggml_tensor_overhead() + ggml_nbytes(result),
        nullptr,
        false,
    });

    struct ggml_tensor *result_copy = ggml_dup_tensor(result_ctx, result);
    memcpy(result_copy->data, result->data, ggml_nbytes(result));

    ggml_free(ctx);  // Free computation context
    return result_copy;  // Return safe copy
}

phonemizer_model phonemizer_load(const std::string &fname) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    phonemizer_model model;

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

    model.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
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

    model.hparams.languages = gguf_get_val_str(ctx, gguf_find_key(ctx, "languages"));
    model.hparams.encoder_vocab_size = gguf_get_val_i32(ctx, gguf_find_key(ctx, "encoder_vocab_size"));
    model.hparams.encoder_symbols = gguf_get_val_str(ctx, gguf_find_key(ctx, "encoder_symbols"));
    model.hparams.decoder_vocab_size = gguf_get_val_i32(ctx, gguf_find_key(ctx, "decoder_vocab_size"));
    model.hparams.decoder_symbols = gguf_get_val_str(ctx, gguf_find_key(ctx, "decoder_symbols"));
    model.hparams.char_repeats = gguf_get_val_i32(ctx, gguf_find_key(ctx, "char_repeats"));
    model.hparams.lowercase = gguf_get_val_bool(ctx, gguf_find_key(ctx, "lowercase"));
    model.hparams.d_model = gguf_get_val_i32(ctx, gguf_find_key(ctx, "d_model"));
    model.hparams.layers = gguf_get_val_i32(ctx, gguf_find_key(ctx, "layers"));
    model.hparams.heads = gguf_get_val_i32(ctx, gguf_find_key(ctx, "heads"));

    gguf_free(ctx);

    return model;
}

void phonemizer_free(phonemizer_model *model) {
    if (model->ctx) {
        ggml_free(model->ctx);
        model->ctx = nullptr;
    }
    if (model->backend) {
        ggml_backend_free(model->backend);
        model->backend = nullptr;
    }
    model->tensors.clear();
    if (model->buffer) {
        ggml_backend_buffer_free(model->buffer);
        model->buffer = nullptr;
    }
}
