#include "data_loading/open_webtext_data_loader.hpp"
#include <iomanip>
#include <iostream>

using namespace tnn;

int main(int argc, char **argv) {
  std::string file_path = "data/open-web-text/train.bin";
  if (argc >= 2) {
    file_path = argv[1];
  }

  std::cout << "Attempting to load: " << file_path << std::endl;

  size_t context_length = 256;
  OpenWebTextDataLoader<float> loader(context_length);

  if (!loader.load_data(file_path)) {
    std::cerr << "Failed to load data. Make sure '" << file_path << "' exists." << std::endl;
    std::cerr << "Run 'python python/openwebtext.py' to generate the data." << std::endl;
    return 1;
  }

  std::cout << "Data loaded successfully." << std::endl;
  std::cout << "Total samples available: " << loader.size() << std::endl;

  Tensor<float> batch_data;
  Tensor<float> batch_labels;
  size_t batch_size = 1;

  // Get the first batch
  if (loader.get_batch(batch_size, batch_data, batch_labels)) {
    std::cout << "\nFirst " << context_length << " tokens (IDs):" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    for (size_t i = 0; i < context_length; ++i) {
      // Data is stored as float in the Tensor, cast to int for display
      int token_id = static_cast<int>(batch_data(0, i));
      std::cout << token_id << " ";

      // formatting for readability
      if ((i + 1) % 16 == 0)
        std::cout << "\n";
    }
    std::cout << "\n------------------------------------------" << std::endl;
    std::cout << "Note: These are BPE token IDs, not raw characters." << std::endl;
  }

  return 0;
}
