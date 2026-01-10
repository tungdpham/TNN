#include "data_loading/open_webtext_data_loader.hpp"
#include "tokenizer/tokenizer.hpp"
#include <iostream>

using namespace tnn;

int main(int argc, char **argv) {
  std::string file_path = "data/open-web-text/train.bin";
  std::string vocab_path = "data/open-web-text/vocab.bin";

  if (argc >= 2) {
    file_path = argv[1];
  }

  std::ifstream f(vocab_path.c_str());
  if (!f.good()) {
    std::cerr << "Vocabulary file not found at " << vocab_path << "\n";
    return 1;
  }

  std::cout << "Attempting to load: " << file_path << std::endl;

  size_t context_length = 1024;
  OpenWebTextDataLoader<float> loader(context_length);

  if (!loader.load_data(file_path)) {
    std::cerr << "Failed to load data. Make sure '" << file_path << "' exists." << std::endl;
    return 1;
  }
  std::cout << "Data loaded successfully." << std::endl;

  Tokenizer tokenizer;
  if (!tokenizer.load(vocab_path)) {
    return 1;
  }
  std::cout << "Tokenizer loaded." << std::endl;

  std::cout << "Total samples available: " << loader.size() << std::endl;

  Tensor<float> batch_data;
  Tensor<float> batch_labels;
  size_t batch_size = 16;

  // Get the first batch
  if (loader.get_batch(batch_size, batch_data, batch_labels)) {
    std::vector<int> tokens;
    for (size_t i = 0; i < context_length; ++i) {
      int token_id = static_cast<int>(batch_data(0, i));
      tokens.push_back(token_id);
      std::cout << token_id << " ";

      if ((i + 1) % 16 == 0)
        std::cout << "\n";
    }

    std::string decoded_text = tokenizer.decode(tokens);
    std::cout << "Decoded Text:\n";
    std::cout << decoded_text << "\n";
  }

  return 0;
}
