#include "ggml.h"
#include "phonemizer.h"
#include "ggml-backend.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <limits.h>
#include <inttypes.h>

/**
 * @brief Performs the forward pass of a phonemizer_model.
 *
 * This function performs the forward pass of a phonemizer_model by processing the input tensor
 * through the embedding, positional encoding, transformer encoder, and fully connected layers.
 * It creates a new graph, adds the operations to the graph, computes the result,
 * and returns it.
 *
 * @param input_tensor The input tensor.
 * @param ctx The ggml_context.
 * @param model The phonemizer_model.
 * @return The result tensor.
 */
struct ggml_tensor *forward(ggml_tensor *input_tensor, struct ggml_context *ctx, const phonemizer_model &model)
{
    struct ggml_cgraph *gf = ggml_new_graph(ctx);

    struct ggml_tensor *x = input_tensor;

    // Embedding
    x = get_embedding(ctx, model.tensors.at("embedding.weight"), x);
    
    // Positional Encoding
    x = get_positional_encoding(ctx, x, model.hparams.d_model, model.hparams.dropout);

    // Transformer Encoder
    for (int i = 0; i < model.hparams.layers; ++i)
    {
        std::string layer_prefix = "encoder.layers." + std::to_string(i);
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

    // Fully Connected Layer
    x = get_linear(ctx, model.tensors.at("fc_out.weight"), model.tensors.at("fc_out.bias"), x);

    ggml_set_name(x, "result");
    ggml_build_forward_expand(gf, x);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    return x;
}

/**
 * Computes the result of a given model on the input data.
 *
 * @param model The phonemizer_model representing the model.
 * @param input The input data as a vector of floats.
 * @return The computed result as a ggml_tensor pointer.
 */
struct ggml_tensor *compute(const phonemizer_model &model, const std::vector<float> &input)
{
    int32_t vocab_size = model.hparams.encoder_vocab_size; // Adjust shape based on input data

    static size_t buf_size = vocab_size * sizeof(float) * 1024 * 1024;
    static void *buf = malloc(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf,
        /*.no_alloc   =*/false,
    };
    struct ggml_context *ctx = ggml_init(params);
    
    // Create input tensor
    struct ggml_tensor *input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
    memcpy(input_tensor->data, input.data(), ggml_nbytes(input_tensor));
    ggml_set_name(input_tensor, "input_tensor");

    memcpy(input_tensor->data, input.data(), ggml_nbytes(input_tensor));

    struct ggml_tensor *result = forward(input_tensor, ctx, model);

    // ggml_graph_print(gf);

    ggml_free(ctx);
    return result;
}

void load_model(const std::string &fname, phonemizer_model &model)
{
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    struct ggml_context *meta = NULL;

    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context *ctx = gguf_init_from_file(fname.c_str(), gguf_params);

    if (!ctx)
    {
        throw std::runtime_error(format("%s: failed to open '%s'\n", __func__, fname.c_str()));
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n", __func__, n_kv, n_tensors, fname.c_str());

    // Calculate required memory size
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

    // Consider extra memory for operations and overhead
    model_size += 10 * 1024 * 1024;  // Adding an extra 10 MB as a buffer

    // Init model backend - currently CPU only
    model.backend = ggml_backend_cpu_init();
    if (!model.backend)
    {
        throw std::runtime_error(format("%s: ggml_backend_cpu_init() failed\n", __func__));
    }

    // Init model context with increased memory size
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ model_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    model.ctx = ggml_init(ggml_params);

    if (!model.ctx)
    {
        throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
    }

    // Add tensors to context
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *cur = ggml_dup_tensor(model.ctx, ggml_get_tensor(meta, name));
        ggml_set_name(cur, name);
        model.tensors[name] = cur;
    }

    // Allocate model buffer
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
            // for the CPU and Metal backend, we can read directly into the tensor
            fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
        }
    }
    fin.close();

    // Load hparams and weights into model params
    model.hparams.encoder_vocab_size = get_i32(ctx, "encoder_vocab_size");
    model.hparams.decoder_vocab_size = get_i32(ctx, "decoder_vocab_size");
    model.hparams.d_model = get_i32(ctx, "d_model");
    model.hparams.d_fft = get_i32(ctx, "d_fft");
    model.hparams.layers = get_i32(ctx, "layers");
    model.hparams.dropout = get_f32(ctx, "dropout");
    model.hparams.heads = get_i32(ctx, "heads");
    
    // Loading weights from the model context
    for (auto &entry : model.tensors)
    {
        const std::string &name = entry.first;
        entry.second = ggml_get_tensor(model.ctx, name.c_str());
    }

    gguf_free(ctx);  // Ensure to free the gguf context
}
