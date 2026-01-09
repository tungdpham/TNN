#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "device/device_manager.hpp"
#include "nn/loss.hpp"
#include "nn/optimizers.hpp"
#include "nn/schedulers.hpp"
#include "nn/sequential.hpp"
#include "nn/train.hpp"
#include "utils/env.hpp"

using namespace tnn;
using namespace std;

// A simple character-level data loader for GPT training
class CharTextDataLoader : public BaseDataLoader<float> {
private:
  string text_;
  map<char, int> char_to_idx_;
  map<int, char> idx_to_char_;
  size_t seq_len_;
  size_t batch_size_;
  size_t cursor_ = 0;
  vector<int> data_indices_;

public:
  CharTextDataLoader(const string &text, size_t seq_len, size_t batch_size)
      : text_(text), seq_len_(seq_len), batch_size_(batch_size) {

    set<char> unique_chars(text.begin(), text.end());
    int idx = 0;
    for (char c : unique_chars) {
      char_to_idx_[c] = idx;
      idx_to_char_[idx] = c;
      idx++;
    }

    // Create data indices
    for (char c : text) {
      data_indices_.push_back(char_to_idx_[c]);
    }
  }

  size_t vocab_size() const { return char_to_idx_.size(); }

  int char_to_idx(char c) {
    if (char_to_idx_.find(c) == char_to_idx_.end())
      return 0;
    return char_to_idx_.at(c);
  }

  char idx_to_char(int idx) {
    if (idx_to_char_.find(idx) == idx_to_char_.end())
      return '?';
    return idx_to_char_.at(idx);
  }

  bool load_data(const string & /*source*/) override {
    cursor_ = 0;
    return true;
  }

  // Returns (B, 1, T, 1) inputs and (B, V, T, 1) one-hot targets
  bool get_next_batch(Tensor<float> &batch_data, Tensor<float> &batch_labels) override {
    if (cursor_ + batch_size_ * seq_len_ + 1 > data_indices_.size()) {
      cursor_ = 0;
      return false;
    }

    size_t V = vocab_size();
    // NCHW format
    // Data: (Batch, 1, SeqLen, 1)
    // Labels: (Batch, Vocab, SeqLen, 1)
    batch_data.resize({batch_size_, 1, seq_len_, 1});
    batch_labels.resize({batch_size_, V, seq_len_, 1});

    float *data_ptr = batch_data.data();
    float *label_ptr = batch_labels.data();
    fill(label_ptr, label_ptr + batch_labels.size(), 0.0f);

    for (size_t b = 0; b < batch_size_; ++b) {
      for (size_t t = 0; t < seq_len_; ++t) {
        size_t global_idx = cursor_ + b * seq_len_ + t;
        if (global_idx + 1 >= data_indices_.size())
          break;

        int input_idx = data_indices_[global_idx];
        int target_idx = data_indices_[global_idx + 1];

        // Input: NCHW index (b, 0, t, 0) -> b * SeqLen + t
        data_ptr[b * seq_len_ + t] = static_cast<float>(input_idx);

        // Target: NCHW index (b, target_idx, t, 0)
        // Offset: b * (V * SeqLen) + target_idx * SeqLen + t
        label_ptr[b * V * seq_len_ + target_idx * seq_len_ + t] = 1.0f;
      }
    }

    cursor_ += batch_size_ * seq_len_;
    return true;
  }
  bool get_batch(size_t batch_size, Tensor<float> &batch_data,
                 Tensor<float> &batch_labels) override {
    size_t old_bs = batch_size_;
    batch_size_ = batch_size;
    bool res = get_next_batch(batch_data, batch_labels);
    batch_size_ = old_bs;
    return res;
  }

  vector<size_t> get_data_shape() const override { return {1, seq_len_, 1}; }
  size_t size() const override { return data_indices_.size(); }
  size_t num_batches() const override {
    return (data_indices_.size() - 1) / (batch_size_ * seq_len_);
  }
  void shuffle() override { /* Not implemented for simplicity */ }
  void reset() override { cursor_ = 0; }
};

int main() {
  // Example text (Shakespeare Sonnet 1)
  string text = "From fairest creatures we desire increase,\n"
                "That thereby beauty's rose might never die,\n"
                "But as the riper should by time decease,\n"
                "His tender heir might bear his memory:\n"
                "But thou, contracted to thine own bright eyes,\n"
                "Feed'st thy light's flame with self-substantial fuel,\n"
                "Making a famine where abundance lies,\n"
                "Thyself thy foe, to thy sweet self too cruel.\n"
                "Thou that art now the world's fresh ornament\n"
                "And only herald to the gaudy spring,\n"
                "Within thine own bud buriest thy content\n"
                "And, tender churl, mak'st waste in niggarding.\n"
                "Pity the world, or else this glutton be,\n"
                "To eat the world's due, by the grave and thee.\n";

  // Repeat text to have enough data for a few batches
  string base_text = text;
  for (int i = 0; i < 10000; ++i)
    text += base_text;

  size_t seq_len = 64;
  size_t batch_size = 32;
  size_t embed_dim = 64;
  size_t num_heads = 4;
  size_t ffn_dim = 256;
  size_t vocab_size = 0;

  CharTextDataLoader loader(text, seq_len, batch_size);
  vocab_size = loader.vocab_size();

  cout << "Text length: " << text.length() << "\n";
  cout << "Vocab size: " << vocab_size << "\n";
  cout << "Num batches: " << loader.num_batches() << "\n";

  SequentialBuilder<float> builder("MiniGPT");
  builder
      .input({1, seq_len, 1})
      // 1. Token Embedding
      .embedding(vocab_size, embed_dim, "token_embed")
      // 2. Positional Embedding
      .positional_embedding(embed_dim, seq_len, "pos_embed")
      .dropout(0.1)
      // 3. Transformer Block
      .gpt_block(embed_dim, num_heads, ffn_dim, 0.1, "gelu")
      .gpt_block(embed_dim, num_heads, ffn_dim, 0.1, "gelu")
      // 4. Final Norm
      .layernorm(1e-5f, true, "ln_f")
      // 5. Projection to Vocab
      .conv2d(vocab_size, 1, 1, 1, 1, 0, 0, true, "head");

  auto model = builder.build();

  // Use GPU if available
  string device_str = Env::get<string>("DEVICE_TYPE", "CPU");
  DeviceType device_type = (device_str == "GPU") ? DeviceType::GPU : DeviceType::CPU;
  cout << "Using device type: " << (device_type == DeviceType::CPU ? "CPU" : "GPU") << endl;
  model.set_device(device_type);
  model.initialize();

  cout << "Model built successfully.\n";
  model.print_summary({16, 1, seq_len, 1});

  // Training Setup
  auto optimizer = OptimizerFactory<float>::create_adam(0.001f);
  auto criterion = LossFactory<float>::create_logsoftmax_crossentropy();
  optimizer->attach(model.parameters(), model.gradients());

  // Training Loop
  int epochs = 3;
  Tensor<float> input, target;
  Tensor<float> output;
  Tensor<float> loss_grad;
  Tensor<float> grad_input;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    loader.reset();
    int batch_idx = 0;
    float total_loss = 0;

    while (loader.get_next_batch(input, target)) {
      // Forward
      model.forward(input, output);

      // Computes Loss
      float loss_val = 0;
      criterion->compute_loss(output, target, loss_val);
      total_loss += loss_val;

      // Backward
      criterion->compute_gradient(output, target, loss_grad);

      model.backward(loss_grad, grad_input);

      // Update
      optimizer->update();
      optimizer->clear_gradients();

      batch_idx++;
    }

    cout << "Epoch " << epoch + 1 << " Loss: " << total_loss / batch_idx << "\n";
  }

  // Text Generation Demo
  cout << "\nGeneration:\n";
  string prompt = "From fairest";
  cout << "Prompt: " << prompt << " -> ";

  // Encode prompt
  vector<int> context;
  for (char c : prompt)
    context.push_back(loader.char_to_idx(c));

  // Generate
  model.set_training(false); // Eval mode

  for (int i = 0; i < 100; ++i) {
    int start = 0;
    if (context.size() > seq_len)
      start = context.size() - seq_len;

    size_t curr_seq_len = context.size() - start;

    Tensor<float> gen_input({1, 1, seq_len, 1}, &getCPU());

    for (size_t t = 0; t < curr_seq_len; ++t) {
      gen_input(0, 0, t, 0) = static_cast<float>(context[start + t]);
    }

    model.forward(gen_input, output);
    // output: (1, V, seq_len, 1)

    size_t last_token_idx = curr_seq_len - 1;

    float max_val = -1e9;
    int max_char_idx = 0;

    Tensor<float> cpu_full_out = output.to_device(&getCPU());
    const float *out_ptr = cpu_full_out.data();

    // Access (0, v, last_token_idx, 0)

    for (size_t v = 0; v < vocab_size; ++v) {
      size_t idx = v * seq_len + last_token_idx;

      float val = out_ptr[idx];
      if (val > max_val) {
        max_val = val;
        max_char_idx = v;
      }
    }

    char next_char = loader.idx_to_char(max_char_idx);
    cout << next_char << flush;
    context.push_back(max_char_idx);
  }
  cout << "\n";

  return 0;
}
