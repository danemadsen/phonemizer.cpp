#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

struct phonemizer_model_hparams
{
    int32_t encoder_vocab_size = 1;
    int32_t decoder_vocab_size = 1;
    int32_t d_model = 512;
    int32_t d_fft = 1024;
    int32_t layers = 4;
    float dropout = 0.1;
    int32_t heads = 1;
};

struct phonemizer_model
{
    phonemizer_model_hparams hparams;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer_w;

    // weights
    struct ggml_tensor *weights;
    struct ggml_tensor *bias;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context *ctx;

    std::map<std::string, struct ggml_tensor *> tensors;

    ggml_backend_buffer_t buffer;
};

void load_model(const std::string &fname, phonemizer_model &model);

struct ggml_tensor *compute(struct phonemizer_model *model, const std::vector<int64_t> &input);