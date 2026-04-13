/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/dnnl_conv2d_ops.hpp"

#ifdef USE_DNNL
#include <cstring>
#include <dnnl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <stdexcept>
#include <unordered_map>

#include "nn/layers_impl/common/conv2d.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dnnl_conv2d {

struct dnnlHandle_t {
  dnnl::engine engine;
  dnnl::stream stream;

  dnnl::convolution_forward fwd_conv;
  dnnl::convolution_backward_data dgrad_conv;
  dnnl::convolution_backward_weights wgrad_conv;

  bool fwd_src_reorder_needed = false;
  dnnl::reorder fwd_src_reorder;
  dnnl::memory fwd_user_src_mem, fwd_packed_src_mem;

  bool fwd_w_reorder_needed = false;
  dnnl::reorder fwd_w_reorder;
  dnnl::memory fwd_user_w_mem, fwd_packed_w_mem;

  bool fwd_dst_reorder_needed = false;
  dnnl::reorder fwd_dst_reorder;
  dnnl::memory fwd_user_dst_mem, fwd_packed_dst_mem;

  dnnl::memory fwd_bias_mem;
  dnnl::memory fwd_scratchpad_mem;

  bool dgrad_dst_reorder_needed = false;
  dnnl::reorder dgrad_dst_reorder;
  dnnl::memory dgrad_user_dst_mem, dgrad_packed_dst_mem;

  bool dgrad_w_reorder_needed = false;
  dnnl::reorder dgrad_w_reorder;
  dnnl::memory dgrad_user_w_mem, dgrad_packed_w_mem;

  bool dgrad_src_reorder_needed = false;  // Note: This is an output
  dnnl::reorder dgrad_src_reorder;
  dnnl::memory dgrad_user_src_mem, dgrad_packed_src_mem;

  dnnl::memory dgrad_scratchpad_mem;

  bool wgrad_src_reorder_needed = false;
  dnnl::reorder wgrad_src_reorder;
  dnnl::memory wgrad_user_src_mem, wgrad_packed_src_mem;

  bool wgrad_dst_reorder_needed = false;
  dnnl::reorder wgrad_dst_reorder;
  dnnl::memory wgrad_user_dst_mem, wgrad_packed_dst_mem;

  bool wgrad_w_reorder_needed = false;  // Note: This is an output
  dnnl::reorder wgrad_w_reorder;
  dnnl::memory wgrad_user_w_mem, wgrad_packed_w_mem;

  dnnl::memory wgrad_bias_mem;
  dnnl::memory wgrad_scratchpad_mem;
};

dnnl::memory::data_type get_dnnl_dtype(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP32:
      return dnnl::memory::data_type::f32;
    case DType_t::FP16:
      return dnnl::memory::data_type::f16;
    case DType_t::BF16:
      return dnnl::memory::data_type::bf16;
    default:
      throw std::runtime_error("dnnl_conv2d: unsupported dtype for DNNL convolution");
  }
}

dnnlHandle_t *initialize_dnnl_handle(ConvolutionStats &stats, DType_t dtype) {
  auto *handle = new dnnlHandle_t();
  handle->engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  handle->stream = dnnl::stream(handle->engine);

  auto dt = get_dnnl_dtype(dtype);

  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t ic = static_cast<int64_t>(stats.in_channels);
  const int64_t ih = static_cast<int64_t>(stats.input_h);
  const int64_t iw = static_cast<int64_t>(stats.input_w);
  const int64_t oc = static_cast<int64_t>(stats.out_channels);
  const int64_t kh = static_cast<int64_t>(stats.kernel_h);
  const int64_t kw = static_cast<int64_t>(stats.kernel_w);
  const int64_t oh = static_cast<int64_t>(stats.output_h);
  const int64_t ow = static_cast<int64_t>(stats.output_w);
  const int64_t sh = static_cast<int64_t>(stats.stride_h);
  const int64_t sw = static_cast<int64_t>(stats.stride_w);
  const int64_t ph = static_cast<int64_t>(stats.pad_h);
  const int64_t pw = static_cast<int64_t>(stats.pad_w);

  // Reference User Definitions
  auto user_src_md = dnnl::memory::desc({n, ic, ih, iw}, dt, dnnl::memory::format_tag::nhwc);
  auto user_weights_md = dnnl::memory::desc({oc, ic, kh, kw}, dt, dnnl::memory::format_tag::ohwi);
  auto user_dst_md = dnnl::memory::desc({n, oc, oh, ow}, dt, dnnl::memory::format_tag::nhwc);

  dnnl::memory::dims strides = {sh, sw};
  dnnl::memory::dims pad_l = {ph, pw};
  dnnl::memory::dims pad_r = {ph, pw};

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dnnl::convolution_forward::primitive_desc fwd_pd;

  // fwd setup
  {
    auto any_src_md = dnnl::memory::desc({n, ic, ih, iw}, dt, dnnl::memory::format_tag::nhwc);
    auto any_weights_md = dnnl::memory::desc({oc, ic, kh, kw}, dt, dnnl::memory::format_tag::any);
    auto any_dst_md = dnnl::memory::desc({n, oc, oh, ow}, dt, dnnl::memory::format_tag::nhwc);

    if (stats.use_bias) {
      auto bias_md = dnnl::memory::desc({oc}, dt, dnnl::memory::format_tag::a);
      fwd_pd = dnnl::convolution_forward::primitive_desc(
          handle->engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto,
          any_src_md, any_weights_md, bias_md, any_dst_md, strides, pad_l, pad_r, attr);
    } else {
      fwd_pd = dnnl::convolution_forward::primitive_desc(
          handle->engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto,
          any_src_md, any_weights_md, any_dst_md, strides, pad_l, pad_r, attr);
    }
    handle->fwd_conv = dnnl::convolution_forward(fwd_pd);
    stats.fwd_workspace_size = static_cast<size_t>(fwd_pd.scratchpad_desc().get_size());

    // src
    handle->fwd_user_src_mem = dnnl::memory(user_src_md, handle->engine, nullptr);
    handle->fwd_src_reorder_needed = (fwd_pd.src_desc() != user_src_md);
    if (handle->fwd_src_reorder_needed) {
      handle->fwd_packed_src_mem =
          dnnl::memory(fwd_pd.src_desc(), handle->engine);  // allocates buffer
      handle->fwd_src_reorder = dnnl::reorder(handle->fwd_user_src_mem, handle->fwd_packed_src_mem);
    }

    // weights
    handle->fwd_user_w_mem = dnnl::memory(user_weights_md, handle->engine, nullptr);
    handle->fwd_w_reorder_needed = (fwd_pd.weights_desc() != user_weights_md);
    if (handle->fwd_w_reorder_needed) {
      handle->fwd_packed_w_mem = dnnl::memory(fwd_pd.weights_desc(), handle->engine);
      handle->fwd_w_reorder = dnnl::reorder(handle->fwd_user_w_mem, handle->fwd_packed_w_mem);
    }

    // dst
    handle->fwd_user_dst_mem = dnnl::memory(user_dst_md, handle->engine, nullptr);
    handle->fwd_dst_reorder_needed = (fwd_pd.dst_desc() != user_dst_md);
    if (handle->fwd_dst_reorder_needed) {
      handle->fwd_packed_dst_mem = dnnl::memory(fwd_pd.dst_desc(), handle->engine);
      handle->fwd_dst_reorder = dnnl::reorder(handle->fwd_packed_dst_mem, handle->fwd_user_dst_mem);
    }

    if (stats.use_bias) {
      handle->fwd_bias_mem = dnnl::memory(fwd_pd.bias_desc(), handle->engine, nullptr);
    }
    if (stats.fwd_workspace_size > 0) {
      handle->fwd_scratchpad_mem = dnnl::memory(fwd_pd.scratchpad_desc(), handle->engine, nullptr);
    }
  }

  // dgrad setup
  {
    auto any_src_md = dnnl::memory::desc({n, ic, ih, iw}, dt, dnnl::memory::format_tag::nhwc);
    auto any_weights_md = dnnl::memory::desc({oc, ic, kh, kw}, dt, dnnl::memory::format_tag::any);
    auto any_dst_md = dnnl::memory::desc({n, oc, oh, ow}, dt, dnnl::memory::format_tag::nhwc);

    auto bwd_data_pd = dnnl::convolution_backward_data::primitive_desc(
        handle->engine, dnnl::algorithm::convolution_auto, any_src_md, any_weights_md, any_dst_md,
        strides, pad_l, pad_r, fwd_pd, attr);

    handle->dgrad_conv = dnnl::convolution_backward_data(bwd_data_pd);
    stats.dgrad_workspace_size = static_cast<size_t>(bwd_data_pd.scratchpad_desc().get_size());

    handle->dgrad_user_dst_mem = dnnl::memory(user_dst_md, handle->engine, nullptr);
    handle->dgrad_dst_reorder_needed = (bwd_data_pd.diff_dst_desc() != user_dst_md);
    if (handle->dgrad_dst_reorder_needed) {
      handle->dgrad_packed_dst_mem = dnnl::memory(bwd_data_pd.diff_dst_desc(), handle->engine);
      handle->dgrad_dst_reorder =
          dnnl::reorder(handle->dgrad_user_dst_mem, handle->dgrad_packed_dst_mem);
    }

    // weights (input to dgrad)
    handle->dgrad_user_w_mem = dnnl::memory(user_weights_md, handle->engine, nullptr);
    handle->dgrad_w_reorder_needed = (bwd_data_pd.weights_desc() != user_weights_md);
    if (handle->dgrad_w_reorder_needed) {
      handle->dgrad_packed_w_mem = dnnl::memory(bwd_data_pd.weights_desc(), handle->engine);
      handle->dgrad_w_reorder = dnnl::reorder(handle->dgrad_user_w_mem, handle->dgrad_packed_w_mem);
    }

    // diff src (output of dgrad)
    handle->dgrad_user_src_mem = dnnl::memory(user_src_md, handle->engine, nullptr);
    handle->dgrad_src_reorder_needed = (bwd_data_pd.diff_src_desc() != user_src_md);
    if (handle->dgrad_src_reorder_needed) {
      handle->dgrad_packed_src_mem = dnnl::memory(bwd_data_pd.diff_src_desc(), handle->engine);
      handle->dgrad_src_reorder =
          dnnl::reorder(handle->dgrad_packed_src_mem, handle->dgrad_user_src_mem);
    }

    if (stats.dgrad_workspace_size > 0) {
      handle->dgrad_scratchpad_mem =
          dnnl::memory(bwd_data_pd.scratchpad_desc(), handle->engine, nullptr);
    }
  }

  // wgrad setup
  {
    auto any_src_md = dnnl::memory::desc({n, ic, ih, iw}, dt, dnnl::memory::format_tag::nhwc);
    auto any_weights_md = dnnl::memory::desc({oc, ic, kh, kw}, dt, dnnl::memory::format_tag::any);
    auto any_dst_md = dnnl::memory::desc({n, oc, oh, ow}, dt, dnnl::memory::format_tag::nhwc);

    dnnl::convolution_backward_weights::primitive_desc bwd_weights_pd;
    if (stats.use_bias) {
      auto bias_md = dnnl::memory::desc({oc}, dt, dnnl::memory::format_tag::a);
      bwd_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
          handle->engine, dnnl::algorithm::convolution_auto, any_src_md, any_weights_md, bias_md,
          any_dst_md, strides, pad_l, pad_r, fwd_pd, attr);
    } else {
      bwd_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
          handle->engine, dnnl::algorithm::convolution_auto, any_src_md, any_weights_md, any_dst_md,
          strides, pad_l, pad_r, fwd_pd, attr);
    }

    handle->wgrad_conv = dnnl::convolution_backward_weights(bwd_weights_pd);
    stats.wgrad_workspace_size = static_cast<size_t>(bwd_weights_pd.scratchpad_desc().get_size());
    stats.bgrad_workspace_size = 0;  // Bias grad typically calculated inline, workspace covers it

    // src (input to wgrad)
    handle->wgrad_user_src_mem = dnnl::memory(user_src_md, handle->engine, nullptr);
    handle->wgrad_src_reorder_needed = (bwd_weights_pd.src_desc() != user_src_md);
    if (handle->wgrad_src_reorder_needed) {
      handle->wgrad_packed_src_mem = dnnl::memory(bwd_weights_pd.src_desc(), handle->engine);
      handle->wgrad_src_reorder =
          dnnl::reorder(handle->wgrad_user_src_mem, handle->wgrad_packed_src_mem);
    }

    // diff dst (input to wgrad)
    handle->wgrad_user_dst_mem = dnnl::memory(user_dst_md, handle->engine, nullptr);
    handle->wgrad_dst_reorder_needed = (bwd_weights_pd.diff_dst_desc() != user_dst_md);
    if (handle->wgrad_dst_reorder_needed) {
      handle->wgrad_packed_dst_mem = dnnl::memory(bwd_weights_pd.diff_dst_desc(), handle->engine);
      handle->wgrad_dst_reorder =
          dnnl::reorder(handle->wgrad_user_dst_mem, handle->wgrad_packed_dst_mem);
    }

    // diff weights (output of wgrad)
    handle->wgrad_user_w_mem = dnnl::memory(user_weights_md, handle->engine, nullptr);
    handle->wgrad_w_reorder_needed = (bwd_weights_pd.diff_weights_desc() != user_weights_md);
    if (handle->wgrad_w_reorder_needed) {
      handle->wgrad_packed_w_mem = dnnl::memory(bwd_weights_pd.diff_weights_desc(), handle->engine);
      handle->wgrad_w_reorder = dnnl::reorder(handle->wgrad_packed_w_mem, handle->wgrad_user_w_mem);
    }

    if (stats.use_bias) {
      handle->wgrad_bias_mem =
          dnnl::memory(bwd_weights_pd.diff_bias_desc(), handle->engine, nullptr);
    }
    if (stats.wgrad_workspace_size > 0) {
      handle->wgrad_scratchpad_mem =
          dnnl::memory(bwd_weights_pd.scratchpad_desc(), handle->engine, nullptr);
    }
  }

  round_workspace_size(stats);
  return handle;
}

void destroy_dnnl_handle(dnnlHandle_t *handle) { delete handle; }

void run_forward(dnnlHandle_t *handle, const ConvolutionStats &stats, const void *input_data,
                 const void *weight_data, const void *bias_data, void *output_data,
                 void *workspace_data) {
  dnnl::stream &s = handle->stream;

  // prepare inputs
  handle->fwd_user_src_mem.set_data_handle(const_cast<void *>(input_data));
  if (handle->fwd_src_reorder_needed) {
    handle->fwd_src_reorder.execute(s, handle->fwd_user_src_mem, handle->fwd_packed_src_mem);
  }

  handle->fwd_user_w_mem.set_data_handle(const_cast<void *>(weight_data));
  if (handle->fwd_w_reorder_needed) {
    handle->fwd_w_reorder.execute(s, handle->fwd_user_w_mem, handle->fwd_packed_w_mem);
  }

  // prepare output
  handle->fwd_user_dst_mem.set_data_handle(output_data);

  // prepare args
  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC,
       handle->fwd_src_reorder_needed ? handle->fwd_packed_src_mem : handle->fwd_user_src_mem},
      {DNNL_ARG_WEIGHTS,
       handle->fwd_w_reorder_needed ? handle->fwd_packed_w_mem : handle->fwd_user_w_mem},
      {DNNL_ARG_DST,
       handle->fwd_dst_reorder_needed ? handle->fwd_packed_dst_mem : handle->fwd_user_dst_mem},
  };

  if (stats.use_bias && bias_data) {
    handle->fwd_bias_mem.set_data_handle(const_cast<void *>(bias_data));
    args[DNNL_ARG_BIAS] = handle->fwd_bias_mem;
  }

  if (handle->fwd_scratchpad_mem) {
    if (!workspace_data) throw std::runtime_error("dnnl_conv2d run_forward: null workspace");
    handle->fwd_scratchpad_mem.set_data_handle(workspace_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->fwd_scratchpad_mem;
  }

  // auto start_time = std::chrono::high_resolution_clock::now();

  handle->fwd_conv.execute(s, args);

  if (handle->fwd_dst_reorder_needed) {
    handle->fwd_dst_reorder.execute(s, handle->fwd_packed_dst_mem, handle->fwd_user_dst_mem);
  }

  s.wait();
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration_ms =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "DNNL convolution forward pass took " << duration_ms << " ms" << std::endl;
}

void run_dgrad(dnnlHandle_t *handle, const ConvolutionStats &stats, const void *grad_output_data,
               const void *weight_data, void *grad_input_data, void *workspace_data) {
  dnnl::stream &s = handle->stream;

  // prepare inputs
  handle->dgrad_user_dst_mem.set_data_handle(const_cast<void *>(grad_output_data));
  if (handle->dgrad_dst_reorder_needed) {
    handle->dgrad_dst_reorder.execute(s, handle->dgrad_user_dst_mem, handle->dgrad_packed_dst_mem);
  }

  handle->dgrad_user_w_mem.set_data_handle(const_cast<void *>(weight_data));
  if (handle->dgrad_w_reorder_needed) {
    handle->dgrad_w_reorder.execute(s, handle->dgrad_user_w_mem, handle->dgrad_packed_w_mem);
  }

  // prepare output
  handle->dgrad_user_src_mem.set_data_handle(grad_input_data);

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_DIFF_DST, handle->dgrad_dst_reorder_needed ? handle->dgrad_packed_dst_mem
                                                           : handle->dgrad_user_dst_mem},
      {DNNL_ARG_WEIGHTS,
       handle->dgrad_w_reorder_needed ? handle->dgrad_packed_w_mem : handle->dgrad_user_w_mem},
      {DNNL_ARG_DIFF_SRC, handle->dgrad_src_reorder_needed ? handle->dgrad_packed_src_mem
                                                           : handle->dgrad_user_src_mem},
  };

  if (handle->dgrad_scratchpad_mem) {
    handle->dgrad_scratchpad_mem.set_data_handle(workspace_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->dgrad_scratchpad_mem;
  }

  // auto start_time = std::chrono::high_resolution_clock::now();

  handle->dgrad_conv.execute(s, args);

  // unpack output (diff src)
  if (handle->dgrad_src_reorder_needed) {
    handle->dgrad_src_reorder.execute(s, handle->dgrad_packed_src_mem, handle->dgrad_user_src_mem);
  }

  s.wait();

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration_ms =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "DNNL convolution backward data pass took " << duration_ms << " ms" << std::endl;
}

void run_wgrad_and_bgrad(dnnlHandle_t *handle, const ConvolutionStats &stats,
                         const void *input_data, const void *grad_output_data,
                         void *grad_weight_data, void *grad_bias_data, void *workspace_data) {
  dnnl::stream &s = handle->stream;

  // prepare inputs
  handle->wgrad_user_src_mem.set_data_handle(const_cast<void *>(input_data));
  if (handle->wgrad_src_reorder_needed) {
    handle->wgrad_src_reorder.execute(s, handle->wgrad_user_src_mem, handle->wgrad_packed_src_mem);
  }

  handle->wgrad_user_dst_mem.set_data_handle(const_cast<void *>(grad_output_data));
  if (handle->wgrad_dst_reorder_needed) {
    handle->wgrad_dst_reorder.execute(s, handle->wgrad_user_dst_mem, handle->wgrad_packed_dst_mem);
  }

  // prepare output (diff weights)
  handle->wgrad_user_w_mem.set_data_handle(grad_weight_data);

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, handle->wgrad_src_reorder_needed ? handle->wgrad_packed_src_mem
                                                      : handle->wgrad_user_src_mem},
      {DNNL_ARG_DIFF_DST, handle->wgrad_dst_reorder_needed ? handle->wgrad_packed_dst_mem
                                                           : handle->wgrad_user_dst_mem},
      {DNNL_ARG_DIFF_WEIGHTS,
       handle->wgrad_w_reorder_needed ? handle->wgrad_packed_w_mem : handle->wgrad_user_w_mem},
  };

  if (stats.use_bias && grad_bias_data) {
    handle->wgrad_bias_mem.set_data_handle(grad_bias_data);
    args[DNNL_ARG_DIFF_BIAS] = handle->wgrad_bias_mem;
  }

  if (handle->wgrad_scratchpad_mem) {
    if (!workspace_data) throw std::runtime_error("dnnl_conv2d run_bwd_weights: null workspace");
    handle->wgrad_scratchpad_mem.set_data_handle(workspace_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->wgrad_scratchpad_mem;
  }

  // auto start_time = std::chrono::high_resolution_clock::now();

  handle->wgrad_conv.execute(s, args);

  // unpack output (diff weights)
  if (handle->wgrad_w_reorder_needed) {
    handle->wgrad_w_reorder.execute(s, handle->wgrad_packed_w_mem, handle->wgrad_user_w_mem);
  }

  s.wait();

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration_ms =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "DNNL convolution backward weights pass took " << duration_ms << " ms" <<
  // std::endl;
}

}  // namespace dnnl_conv2d
}  // namespace cpu
}  // namespace tnn

#endif