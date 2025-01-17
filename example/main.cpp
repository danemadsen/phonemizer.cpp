#include "phonemizer.h"
#include <string.h>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

int main(void)
{
    ggml_time_init();

    // Load model and run forward
    phonemizer_model model;
    load_model("./model/deep_phonemizer.gguf", model);
    
    // Get the encoder vocabulary size from the model hyperparameters
    int vocab_size = model.hparams.encoder_vocab_size;

    // Create example tensor with valid indices
    std::vector<float> input_data(vocab_size);
    std::iota(input_data.begin(), input_data.end(), 0);

    struct ggml_tensor *result = compute(model, input_data);
    printf("Forward computed\n");

    // Printing
    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    printf("Result: [");
    for (int i = 0; i < result->ne[0]; i++)
    {
        printf("%.2f, ", out_data[i]);
    }
    printf("]\n");

    ggml_free(model.ctx);
    return 0;
}