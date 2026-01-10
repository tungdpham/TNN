#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "data_loading/open_webtext_data_loader.hpp"
#include "nn/example_models.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "tokenizer/tokenizer.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

// Helper to convert sparse indices to one-hot vectors
// Input: (B, T), Output: (B, V, T, 1)
void one_hot_encode(const Tensor<float> &targets, Tensor<float> &one_hot_targets,
                    size_t vocab_size) {
  size_t B = targets.shape()[0];
  size_t T_seq = targets.shape()[1];

  // Only resize if necessary to avoid re-allocation
  if (one_hot_targets.shape() != vector<size_t>{B, vocab_size, T_seq, 1}) {
    one_hot_targets.resize({B, vocab_size, T_seq, 1});
  }

  float *out_ptr = one_hot_targets.data();
  const float *in_ptr = targets.data();

  // Fill with zeros
  std::fill(out_ptr, out_ptr + one_hot_targets.size(), 0.0f);

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T_seq; ++t) {
      float token_id = in_ptr[b * T_seq + t];
      int idx = static_cast<int>(token_id);
      if (idx >= 0 && idx < (int)vocab_size) {
        // Layout is NCHW where C is vocab_size, H is seq_len, W is 1
        // Index: (b, idx, t, 0)
        // Offset = b * (C * H * W) + idx * (H * W) + t * W + 0
        //        = b * (V * T) + idx * T + t
        size_t offset = b * (vocab_size * T_seq) + idx * T_seq + t;
        out_ptr[offset] = 1.0f;
      }
    }
  }
}

int main(int argc, char **argv) {
  string data_path = "data/open-web-text/train.bin";
  string vocab_path = "data/open-web-text/vocab.bin";

  if (argc >= 2) {
    data_path = argv[1];
  }
  if (argc >= 3) {
    vocab_path = argv[2];
  }

  if (!std::filesystem::exists(data_path)) {
    cerr << "Data file not found: " << data_path << endl;
    cerr << "Please run python/openwebtext.py first to generate data." << endl;
    return 1;
  }

  Tokenizer tokenizer;
  if (!tokenizer.load(vocab_path)) {
    cerr << "Failed to load vocab from: " << vocab_path << endl;
    return 1;
  }

  size_t vocab_size = tokenizer.vocab_size();
  cout << "Using Vocab Size: " << vocab_size << endl;

  string device_str = Env::get<string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;

  size_t batch_size = 2;
  size_t seq_len = 512;

  cout << "Device: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;
  cout << "Batch Size: " << batch_size << endl;
  cout << "Seq Len: " << seq_len << endl;

  OpenWebTextDataLoader<float> loader(seq_len);
  if (!loader.load_data(data_path)) {
    cerr << "Failed to load data." << endl;
    return 1;
  }
  cout << "Data loaded. Total tokens: " << loader.size() + seq_len << endl;

  auto model = create_gpt2(seq_len, vocab_size);

  model.set_device(device_type);
  model.initialize();

  model.print_summary({batch_size, 1, seq_len, 1});

  auto optimizer = OptimizerFactory<float>::create_adam(0.001f, 0.9f, 0.95f, 1e-8f, 0.1f);

  auto criterion = LossFactory<float>::create_logsoftmax_crossentropy();

  optimizer->attach(model.parameters(), model.gradients());

  size_t max_steps = 1000;
  size_t step = 0;

  Tensor<float> raw_input, raw_target;
  Tensor<float> model_input, one_hot_target;
  Tensor<float> output, loss_grad, grad_input;

  model_input.resize({batch_size, 1, seq_len, 1});

  loader.shuffle();
  loader.reset();

  float total_loss = 0;
  int log_interval = 10;

  while (step < max_steps) {
    if (!loader.get_batch(batch_size, raw_input, raw_target)) {
      loader.reset();
      continue;
    }

    float *raw_in_ptr = raw_input.data();
    float *mod_in_ptr = model_input.data();
    std::copy(raw_in_ptr, raw_in_ptr + raw_input.size(), mod_in_ptr);

    one_hot_encode(raw_target, one_hot_target, vocab_size);

    model.forward(model_input, output);

    float loss_val = 0;
    criterion->compute_loss(output, one_hot_target, loss_val);
    total_loss += loss_val;

    criterion->compute_gradient(output, one_hot_target, loss_grad);
    model.backward(loss_grad, grad_input);

    optimizer->update();
    optimizer->clear_gradients();

    step++;
    if (step % log_interval == 0) {
      cout << "Step " << step << " | Loss: " << total_loss / log_interval << endl;
      total_loss = 0;
    }
  }

  model.save_to_file("gpt2_model");
  cout << "Training finished. Model saved to gpt2_model." << endl;

  return 0;
}
