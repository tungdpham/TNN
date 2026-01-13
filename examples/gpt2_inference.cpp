#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "data_loading/open_webtext_data_loader.hpp"
#include "nn/sequential.hpp"
#include "tokenizer/tokenizer.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

int main(int argc, char **argv) {
  string model_path = "model_snapshots/gpt2";
  string vocab_path = "data/open-web-text/vocab.bin";
  string data_path = "data/open-web-text/train.bin";

  cout << "Loading model from: " << model_path << endl;
  cout << "Loading vocab from: " << vocab_path << endl;

  Tokenizer tokenizer;
  if (!tokenizer.load(vocab_path)) {
    cerr << "Failed to load vocab from: " << vocab_path << endl;
    return 1;
  }

  string device_str = Env::get<string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  cout << "Using device: " << (device_type == DeviceType::GPU ? "GPU" : "CPU") << endl;

  auto model = Sequential<float>::from_file(model_path,
                                            device_type == DeviceType::GPU ? &getGPU() : &getCPU());
  model.set_training(false);

  size_t seq_len = 512;

  OpenWebTextDataLoader<float> loader(seq_len);
  if (!loader.load_data(data_path)) {
    cerr << "Could not load data for prompt from: " << data_path << endl;
    return 1;
  }

  Tensor<float> raw_input, raw_target;
  loader.shuffle();
  if (!loader.get_batch(1, raw_input, raw_target)) {
    cerr << "Failed to get a batch from data loader." << endl;
    return 1;
  }

  size_t prompt_len = 30;
  vector<int> current_tokens;
  for (size_t i = 0; i < prompt_len; ++i) {
    current_tokens.push_back(static_cast<int>(raw_input(0, i)));
  }

  cout << "\n[PROMPT]: " << tokenizer.decode(current_tokens) << endl;
  cout << "\n[GENERATED]: " << flush;

  size_t num_to_generate = 50;
  for (size_t i = 0; i < num_to_generate; ++i) {
    Tensor<float> model_input({1, seq_len});
    std::fill(model_input.data(), model_input.data() + model_input.size(), 0.0f);

    size_t tokens_to_use = std::min(current_tokens.size(), seq_len);
    size_t start_token_idx = current_tokens.size() - tokens_to_use;

    for (size_t j = 0; j < tokens_to_use; ++j) {
      model_input(0, j) = static_cast<float>(current_tokens[start_token_idx + j]);
    }

    Tensor<float> output;
    model.forward(model_input, output);

    // Transfer output to CPU for sampling
    Tensor<float> cpu_output = output.to_cpu();

    size_t vocab_size = tokenizer.vocab_size();
    size_t last_step_idx = tokens_to_use - 1;

    const float *logits = cpu_output.data() + (last_step_idx * vocab_size);

    // Check for NaNs
    if (std::isnan(logits[0])) {
      cerr << "\nModel output contains NaNs. The model likely diverged during training." << endl;
      return 1;
    }

    // Greedy sampling: argmax
    int next_token = 0;
    float max_logit = logits[0];
    for (int v = 1; v < (int)vocab_size; ++v) {
      if (logits[v] > max_logit) {
        max_logit = logits[v];
        next_token = v;
      }
    }

    current_tokens.push_back(next_token);
    string decoded = tokenizer.decode(next_token);
    cout << decoded << flush;

    if (next_token == 50256)
      break;
  }

  cout << "\n\n[FULL TEXT]:\n" << tokenizer.decode(current_tokens) << endl;

  return 0;
}
