#include "ggml.h"
#include "ggml-backend.h"

struct phonemizer_model;

struct phonemizer_model * phonemizer_load(const char * fname);

void phonemizer_free(struct phonemizer_model * model);

#ifdef __cplusplus
#include <string>
#include <vector>

std::vector<std::string> phonemize(const char *text, struct phonemizer_model *model);

class Phonemizer {
    private:
        struct phonemizer_model *model;
    
    public:
        Phonemizer(const char *model_path);

        ~Phonemizer();

        std::vector<std::string> operator()(const std::string& text) const;
};

#endif