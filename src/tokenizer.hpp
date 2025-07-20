#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class SequenceTokenizer {
    private:
        const std::string pad_token = " ";
        const std::string end_token = "<end>";

        std::vector<std::string> tokens;
        std::vector<std::string> languages;
        int vocab_length;
        int char_repeats;
        bool lowercase;
        int pad_index;
        int end_index;
        std::unordered_set<std::string> special_tokens;

    public:
        SequenceTokenizer(
            const std::vector<std::string> symbols, 
            const std::vector<std::string> languages, 
            int vocab_length,
            int char_repeats, 
            bool lowercase
        ) : languages(languages),
            vocab_length(vocab_length),
            char_repeats(char_repeats), 
            lowercase(lowercase)
        {
            tokens.push_back(pad_token);
            special_tokens.insert(pad_token);

            for (const auto& lang : languages) {
                std::string lang_token = "<" + lang + ">";
                tokens.push_back(lang_token);
                special_tokens.insert(lang_token);
            }

            tokens.push_back(end_token);
            end_index = tokens.size() - 1;

            for (const auto& symbol : symbols) {
                tokens.push_back(symbol);
            }
        }

        std::vector<int64_t> operator()(
            const std::string sentence, 
            const std::string language = "en_us"
        ) const {
            std::string processed_sentence = sentence;
            if (lowercase) {
                std::transform(processed_sentence.begin(), processed_sentence.end(), processed_sentence.begin(), ::tolower);
            }

            std::vector<int64_t> sequence;
            for (char c : processed_sentence) {
                std::string symbol(1, c);
                auto index = get_token(symbol);
                if (index == -1) continue; // skip unknown symbols
                for (int i = 0; i < char_repeats; ++i) {
                    sequence.push_back(index);
                }
            }

            auto index = get_token("<" + language + ">");
            sequence.insert(sequence.begin(), index);
            sequence.push_back(end_index);

            while (sequence.size() < vocab_length) {
                sequence.push_back(pad_index);
            }

            if (sequence.size() > vocab_length) {
                sequence.resize(vocab_length);
            }

            return sequence;
        }

        std::vector<std::string> decode(const std::vector<int64_t>& sequence) const {
            std::vector<int64_t> processed_sequence;

            processed_sequence.push_back(sequence.front());
            for (size_t i = 1; i < sequence.size() - 1; i += char_repeats) {
                processed_sequence.push_back(sequence[i]);
            }
            processed_sequence.push_back(sequence.back());

            std::vector<std::string> decoded;
            for (int64_t token : processed_sequence) {
                if (token == end_index) {
                    break;
                }
                decoded.push_back(tokens[token]);
            }

            return decoded;
        }
    
        std::vector<int64_t> clean(const std::vector<int64_t>& sequence) const {
            std::vector<int64_t> processed_sequence = sequence;

            // remove all special tokens from the sequence
            for (auto token : special_tokens) {
                auto special_token_index = get_token(token);
                if (special_token_index != -1) {
                    processed_sequence.erase(std::remove(processed_sequence.begin(), processed_sequence.end(), special_token_index), processed_sequence.end());
                }
            }

            // extract everything between the start and end tokens
            auto end = std::find(processed_sequence.begin(), processed_sequence.end(), end_index);
            if (end != processed_sequence.end()) {
                processed_sequence.erase(end, processed_sequence.end());
            }

            // Remove consecutive duplicate tokens
            auto last = std::unique(processed_sequence.begin(), processed_sequence.end());
            processed_sequence.erase(last, processed_sequence.end());

            return processed_sequence;
        }

        int64_t get_token(const std::string& token) const {
            auto it = std::find(tokens.begin(), tokens.end(), token);

            if (it != tokens.end()) {
                return std::distance(tokens.begin(), it);
            }

            return -1;
        }
};

#endif