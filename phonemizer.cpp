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
 * @brief Loads the weights of a module from the ggml_context.
 *
 * This function loads the weights of a module from the ggml_context by retrieving
 * the tensors with the specified names and assigning them to the corresponding
 * fields of the module.
 *
 * @param model The module to load the weights into.
 */
void load_weights(module &model)
{
    // Loading weights from the model context
    for (auto &entry : model.tensors)
    {
        const std::string &name = entry.first;
        entry.second = ggml_get_tensor(model.ctx, name.c_str());
    }
}

/**
 * @brief Loads the hyperparameters of a module from the gguf_context.
 *
 * This function loads the hyperparameters of a module from the gguf_context by
 * retrieving the values of the specified keys and assigning them to the corresponding
 * fields of the module's hparams struct.
 *
 * @param model The module to load the hyperparameters into.
 * @param ctx The gguf_context.
 */
void load_hparams(module &model, gguf_context *ctx)
{
    auto &hparams = model.hparams;
    hparams.encoder_vocab_size = get_i32(ctx, "encoder_vocab_size");
    hparams.decoder_vocab_size = get_i32(ctx, "decoder_vocab_size");
    hparams.d_model = get_i32(ctx, "d_model");
    hparams.d_fft = get_i32(ctx, "d_fft");
    hparams.layers = get_i32(ctx, "layers");
    hparams.dropout = get_f32(ctx, "dropout");
    hparams.heads = get_i32(ctx, "heads");
    printf("%s: encoder_vocab_size = %d, decoder_vocab_size = %d, d_model = %d, d_fft = %d, layers = %d, dropout = %f, heads = %d\n",
           __func__, hparams.encoder_vocab_size, hparams.decoder_vocab_size, hparams.d_model, hparams.d_fft, hparams.layers, hparams.dropout, hparams.heads);
}

/**
 * @brief Creates an input tensor from a vector of floats.
 *
 * This function creates a 1-dimensional input tensor from a vector of floats.
 * It allocates memory for the tensor, copies the input data into the tensor,
 * and sets the tensor's name.
 *
 * @param input The input data as a vector of floats.
 * @param ctx The ggml_context.
 * @param shape The shape of the tensor.
 * @return A pointer to the created input tensor.
 */
struct ggml_tensor *create_input_tensor(const std::vector<float> &input, struct ggml_context *ctx, int32_t shape)
{
    struct ggml_tensor *input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape);
    memcpy(input_tensor->data, input.data(), ggml_nbytes(input_tensor));
    ggml_set_name(input_tensor, "input_tensor");

    return input_tensor;
}

/**
 * @brief Performs the forward pass of a module.
 *
 * This function performs the forward pass of a module by processing the input tensor
 * through the embedding, positional encoding, transformer encoder, and fully connected layers.
 * It creates a new graph, adds the operations to the graph, computes the result,
 * and returns it.
 *
 * @param input_tensor The input tensor.
 * @param ctx The ggml_context.
 * @param model The module.
 * @return The result tensor.
 */
struct ggml_tensor *forward(ggml_tensor *input_tensor, struct ggml_context *ctx, const module &model)
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
        x = ggml_transformer_encoder_layer(
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
            model.tensors.at(layer_prefix + ".norm1.weight"),
            model.tensors.at(layer_prefix + ".norm1.bias"),
            model.tensors.at(layer_prefix + ".norm2.weight"),
            model.tensors.at(layer_prefix + ".norm2.bias"),
            model.hparams.heads,
            model.hparams.d_model,
            model.hparams.d_fft,
            model.hparams.dropout);
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
 * @param model The module representing the model.
 * @param input The input data as a vector of floats.
 * @return The computed result as a ggml_tensor pointer.
 */
struct ggml_tensor *compute(const module &model, const std::vector<float> &input)
{
    int32_t shape = model.hparams.encoder_vocab_size; // Adjust shape based on input data

    static size_t buf_size = shape * sizeof(float) * 1024 * 1024;
    static void *buf = malloc(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf,
        /*.no_alloc   =*/false,
    };
    struct ggml_context *ctx = ggml_init(params);
    struct ggml_tensor *input_tensor = create_input_tensor(input, ctx, shape);
    struct ggml_tensor *result = forward(input_tensor, ctx, model);

    // ggml_graph_print(gf);

    ggml_free(ctx);
    return result;
}

void load_model(const std::string &fname, module &model)
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

    // Evaluate context size
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

    // Init model backend - currently CPU only
    model.backend = ggml_backend_cpu_init();
    if (!model.backend)
    {
        throw std::runtime_error(format("%s: ggml_backend_cpu_init() failed\n", __func__));
    }

    // Init model context
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ (n_tensors + 1) * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };

    model.ctx = ggml_init(ggml_params);

    if (!model.ctx)
    {
        throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
    }

    // Add tensors to context
    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *t = ggml_get_tensor(meta, name);
        struct ggml_tensor *cur = ggml_dup_tensor(model.ctx, t);
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
    load_hparams(model, ctx);
    load_weights(model);
}
