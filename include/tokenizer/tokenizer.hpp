#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace tnn {

class Tokenizer {
public:
  Tokenizer() = default;

  bool load(const std::string &vocab_path) {
    std::ifstream file(vocab_path, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
      return false;
    }

    uint32_t vocab_size;
    file.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));

    vocab_.resize(vocab_size);

    for (uint32_t i = 0; i < vocab_size; ++i) {
      uint32_t token_len;
      file.read(reinterpret_cast<char *>(&token_len), sizeof(token_len));

      if (token_len > 0) {
        vocab_[i].resize(token_len);
        file.read(&vocab_[i][0], token_len);
      }
    }

    return true;
  }

  std::string decode(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(vocab_.size())) {
      return "<unk>";
    }
    return vocab_[token_id];
  }

  std::string decode(const std::vector<int> &tokens) const {
    std::string result;
    for (int id : tokens) {
      result += decode(id);
    }
    return result;
  }

  // For float tensors (standard in this framework)
  std::string decode(const std::vector<float> &tokens) const {
    std::string result;
    for (float val : tokens) {
      result += decode(static_cast<int>(val));
    }
    return result;
  }

  size_t vocab_size() const { return vocab_.size(); }

private:
  std::vector<std::string> vocab_;
};

}  // namespace tnn
