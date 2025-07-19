#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

typedef struct {
    const char * languages = nullptr;
    int32_t encoder_vocab_size = 1;
    const char * encoder_symbols = nullptr;
    int32_t decoder_vocab_size = 1;
    const char * decoder_symbols = nullptr;
    int8_t char_repeats = 1;
    bool lowercase = true;
    int32_t d_model = 512;
    int32_t layers = 4;
    int32_t heads = 1;
} phonemizer_model_hparams;

typedef struct
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
} phonemizer_model;

phonemizer_model phonemizer_load(const std::string &fname);

void phonemizer_free(phonemizer_model *model);

struct ggml_tensor *compute(phonemizer_model *model, const std::vector<int64_t> &input);