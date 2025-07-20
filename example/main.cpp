#include "phonemizer.h"
#include <string.h>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

int main(void)
{
    ggml_backend_load_all();
    ggml_time_init();

    // Load model and run forward
    struct phonemizer_model * model = phonemizer_load("./model/deep_phonemizer.gguf");

    printf("Model loaded successfully.\n");
    
    auto phonemized_text = phonemize("Hello world", model);

    for (const auto &phoneme : phonemized_text) {
        std::cout << phoneme << " ";
    }
    std::cout << std::endl;

    phonemizer_free(model);
    return 0;
}