/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/dnnl_conv2d_ops.hpp"

#ifdef USE_DNNL

#include <dnnl.hpp>
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
  dnnl::convolution_backward_data bwd_data_conv;
  dnnl::convolution_backward_weights bwd_weights_conv;

  bool use_bias = false;

  dnnl::memory fwd_src_mem;
  dnnl::memory fwd_dst_mem;
  dnnl::memory fwd_bias_mem;
  dnnl::memory fwd_scratchpad_mem;
  bool has_fwd_scratchpad = false;

  dnnl::memory bwd_data_diff_dst_mem;
  dnnl::memory bwd_data_diff_src_mem;
  dnnl::memory bwd_data_scratchpad_mem;
  bool has_bwd_data_scratchpad = false;

  dnnl::memory bwd_weights_src_mem;
  dnnl::memory bwd_weights_diff_dst_mem;
  dnnl::memory bwd_weights_diff_bias_mem;
  dnnl::memory bwd_weights_scratchpad_mem;
  bool has_bwd_weights_scratchpad = false;

  bool needs_weights_reorder = false;
  dnnl::memory::desc user_weights_ohwi_md;
  dnnl::memory user_weights_mem;
  dnnl::memory packed_weights;
  dnnl::memory packed_grad_weights;
  dnnl::memory user_grad_weights_mem;
  dnnl::reorder weights_to_packed_reorder;
  dnnl::reorder grad_weights_to_user_reorder;

  dnnl::memory fwd_weights_direct_mem;
  dnnl::memory bwd_data_weights_direct_mem;
  dnnl::memory bwd_weights_diff_weights_direct_mem;
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
  handle->use_bias = stats.use_bias;

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

  auto src_md = dnnl::memory::desc({n, ic, ih, iw}, dt, dnnl::memory::format_tag::nhwc);
  auto user_weights_ohwi_md_local =
      dnnl::memory::desc({oc, ic, kh, kw}, dt, dnnl::memory::format_tag::ohwi);
  auto weights_md = dnnl::memory::desc({oc, ic, kh, kw}, dt, dnnl::memory::format_tag::any);
  auto dst_md = dnnl::memory::desc({n, oc, oh, ow}, dt, dnnl::memory::format_tag::nhwc);

  dnnl::memory::dims strides = {sh, sw};
  dnnl::memory::dims pad_l = {ph, pw};
  dnnl::memory::dims pad_r = {ph, pw};

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dnnl::convolution_forward::primitive_desc fwd_pd;
  if (stats.use_bias) {
    auto bias_md = dnnl::memory::desc({oc}, dt, dnnl::memory::format_tag::a);
    fwd_pd = dnnl::convolution_forward::primitive_desc(
        handle->engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto,
        src_md, weights_md, bias_md, dst_md, strides, pad_l, pad_r, attr);
  } else {
    fwd_pd = dnnl::convolution_forward::primitive_desc(
        handle->engine, dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_auto,
        src_md, weights_md, dst_md, strides, pad_l, pad_r, attr);
  }

  handle->fwd_conv = dnnl::convolution_forward(fwd_pd);
  stats.fwd_workspace_size = static_cast<size_t>(fwd_pd.scratchpad_desc().get_size());

  handle->fwd_src_mem = dnnl::memory(fwd_pd.src_desc(), handle->engine, nullptr);
  handle->fwd_dst_mem = dnnl::memory(fwd_pd.dst_desc(), handle->engine, nullptr);
  if (stats.use_bias) {
    handle->fwd_bias_mem = dnnl::memory(fwd_pd.bias_desc(), handle->engine, nullptr);
  }
  if (stats.fwd_workspace_size > 0) {
    handle->has_fwd_scratchpad = true;
    handle->fwd_scratchpad_mem = dnnl::memory(fwd_pd.scratchpad_desc(), handle->engine, nullptr);
  }

  dnnl::memory::desc optimal_weights_md = fwd_pd.weights_desc();
  handle->needs_weights_reorder = !(optimal_weights_md == user_weights_ohwi_md_local);
  if (handle->needs_weights_reorder) {
    handle->user_weights_ohwi_md = user_weights_ohwi_md_local;
    handle->packed_weights = dnnl::memory(optimal_weights_md, handle->engine);
    handle->packed_grad_weights = dnnl::memory(optimal_weights_md, handle->engine);
    auto reorder_pd_fwd = dnnl::reorder::primitive_desc(handle->engine, user_weights_ohwi_md_local,
                                                        handle->engine, optimal_weights_md);
    handle->weights_to_packed_reorder = dnnl::reorder(reorder_pd_fwd);
    auto reorder_pd_bwd = dnnl::reorder::primitive_desc(handle->engine, optimal_weights_md,
                                                        handle->engine, user_weights_ohwi_md_local);
    handle->grad_weights_to_user_reorder = dnnl::reorder(reorder_pd_bwd);
    handle->user_weights_mem = dnnl::memory(user_weights_ohwi_md_local, handle->engine, nullptr);
    handle->user_grad_weights_mem =
        dnnl::memory(user_weights_ohwi_md_local, handle->engine, nullptr);
  } else {
    handle->fwd_weights_direct_mem = dnnl::memory(optimal_weights_md, handle->engine, nullptr);
  }

  {
    auto bwd_data_pd = dnnl::convolution_backward_data::primitive_desc(
        handle->engine, dnnl::algorithm::convolution_auto, src_md, optimal_weights_md, dst_md,
        strides, pad_l, pad_r, fwd_pd, attr);

    handle->bwd_data_conv = dnnl::convolution_backward_data(bwd_data_pd);
    stats.dgrad_workspace_size = static_cast<size_t>(bwd_data_pd.scratchpad_desc().get_size());

    handle->bwd_data_diff_dst_mem =
        dnnl::memory(bwd_data_pd.diff_dst_desc(), handle->engine, nullptr);
    handle->bwd_data_diff_src_mem =
        dnnl::memory(bwd_data_pd.diff_src_desc(), handle->engine, nullptr);
    if (stats.dgrad_workspace_size > 0) {
      handle->has_bwd_data_scratchpad = true;
      handle->bwd_data_scratchpad_mem =
          dnnl::memory(bwd_data_pd.scratchpad_desc(), handle->engine, nullptr);
    }
    if (!handle->needs_weights_reorder) {
      handle->bwd_data_weights_direct_mem =
          dnnl::memory(bwd_data_pd.weights_desc(), handle->engine, nullptr);
    }
  }

  {
    dnnl::convolution_backward_weights::primitive_desc bwd_weights_pd;
    if (stats.use_bias) {
      auto bias_md = dnnl::memory::desc({oc}, dt, dnnl::memory::format_tag::a);
      bwd_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
          handle->engine, dnnl::algorithm::convolution_auto, src_md, optimal_weights_md, bias_md,
          dst_md, strides, pad_l, pad_r, fwd_pd, attr);
    } else {
      bwd_weights_pd = dnnl::convolution_backward_weights::primitive_desc(
          handle->engine, dnnl::algorithm::convolution_auto, src_md, optimal_weights_md, dst_md,
          strides, pad_l, pad_r, fwd_pd, attr);
    }

    handle->bwd_weights_conv = dnnl::convolution_backward_weights(bwd_weights_pd);
    stats.wgrad_workspace_size = static_cast<size_t>(bwd_weights_pd.scratchpad_desc().get_size());

    stats.bgrad_workspace_size = 0;

    handle->bwd_weights_src_mem = dnnl::memory(bwd_weights_pd.src_desc(), handle->engine, nullptr);
    handle->bwd_weights_diff_dst_mem =
        dnnl::memory(bwd_weights_pd.diff_dst_desc(), handle->engine, nullptr);
    if (stats.use_bias) {
      handle->bwd_weights_diff_bias_mem =
          dnnl::memory(bwd_weights_pd.diff_bias_desc(), handle->engine, nullptr);
    }
    if (stats.wgrad_workspace_size > 0) {
      handle->has_bwd_weights_scratchpad = true;
      handle->bwd_weights_scratchpad_mem =
          dnnl::memory(bwd_weights_pd.scratchpad_desc(), handle->engine, nullptr);
    }
    if (!handle->needs_weights_reorder) {
      handle->bwd_weights_diff_weights_direct_mem =
          dnnl::memory(bwd_weights_pd.diff_weights_desc(), handle->engine, nullptr);
    }
  }

  round_workspace_size(stats);
  return handle;
}

void destroy_dnnl_handle(dnnlHandle_t *handle) { delete handle; }

void run_forward(dnnlHandle_t *handle, const ConvolutionStats & /*stats*/, const void *input_data,
                 const void *weight_data, const void *bias_data, void *output_data,
                 void *workspace_data) {
  dnnl::stream &s = handle->stream;

  handle->fwd_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->fwd_dst_mem.set_data_handle(output_data);

  dnnl::memory *weights_mem_ptr;
  if (handle->needs_weights_reorder) {
    handle->user_weights_mem.set_data_handle(const_cast<void *>(weight_data));
    handle->weights_to_packed_reorder.execute(
        s, {{DNNL_ARG_FROM, handle->user_weights_mem}, {DNNL_ARG_TO, handle->packed_weights}});
    weights_mem_ptr = &handle->packed_weights;
  } else {
    handle->fwd_weights_direct_mem.set_data_handle(const_cast<void *>(weight_data));
    weights_mem_ptr = &handle->fwd_weights_direct_mem;
  }

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, handle->fwd_src_mem},
      {DNNL_ARG_WEIGHTS, *weights_mem_ptr},
      {DNNL_ARG_DST, handle->fwd_dst_mem},
  };

  if (handle->use_bias && bias_data) {
    handle->fwd_bias_mem.set_data_handle(const_cast<void *>(bias_data));
    args[DNNL_ARG_BIAS] = handle->fwd_bias_mem;
  }

  if (handle->has_fwd_scratchpad) {
    if (!workspace_data) {
      throw std::runtime_error(
          "dnnl_conv2d run_forward: scratchpad required but workspace is null");
    }
    handle->fwd_scratchpad_mem.set_data_handle(workspace_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->fwd_scratchpad_mem;
  }

  handle->fwd_conv.execute(s, args);
  s.wait();
}

void run_backward_data(dnnlHandle_t *handle, const ConvolutionStats & /*stats*/,
                       const void *grad_output_data, const void *weight_data, void *grad_input_data,
                       void *workspace_data) {
  dnnl::stream &s = handle->stream;

  handle->bwd_data_diff_dst_mem.set_data_handle(const_cast<void *>(grad_output_data));
  handle->bwd_data_diff_src_mem.set_data_handle(grad_input_data);

  dnnl::memory *weights_mem_ptr;
  if (handle->needs_weights_reorder) {
    weights_mem_ptr = &handle->packed_weights;
  } else {
    handle->bwd_data_weights_direct_mem.set_data_handle(const_cast<void *>(weight_data));
    weights_mem_ptr = &handle->bwd_data_weights_direct_mem;
  }

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_DIFF_DST, handle->bwd_data_diff_dst_mem},
      {DNNL_ARG_WEIGHTS, *weights_mem_ptr},
      {DNNL_ARG_DIFF_SRC, handle->bwd_data_diff_src_mem},
  };

  if (handle->has_bwd_data_scratchpad) {
    if (!workspace_data) {
      throw std::runtime_error(
          "dnnl_conv2d run_backward_data: scratchpad required but workspace is null");
    }
    handle->bwd_data_scratchpad_mem.set_data_handle(workspace_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->bwd_data_scratchpad_mem;
  }

  handle->bwd_data_conv.execute(s, args);
  s.wait();
}

void run_backward_weights_and_bias(dnnlHandle_t *handle, const ConvolutionStats & /*stats*/,
                                   const void *input_data, const void *grad_output_data,
                                   void *grad_weight_data, void *grad_bias_data,
                                   void *workspace_data) {
  dnnl::stream &s = handle->stream;

  handle->bwd_weights_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->bwd_weights_diff_dst_mem.set_data_handle(const_cast<void *>(grad_output_data));

  dnnl::memory *diff_weights_mem_ptr;
  if (handle->needs_weights_reorder) {
    diff_weights_mem_ptr = &handle->packed_grad_weights;
  } else {
    handle->bwd_weights_diff_weights_direct_mem.set_data_handle(grad_weight_data);
    diff_weights_mem_ptr = &handle->bwd_weights_diff_weights_direct_mem;
  }

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, handle->bwd_weights_src_mem},
      {DNNL_ARG_DIFF_DST, handle->bwd_weights_diff_dst_mem},
      {DNNL_ARG_DIFF_WEIGHTS, *diff_weights_mem_ptr},
  };

  if (handle->use_bias && grad_bias_data) {
    handle->bwd_weights_diff_bias_mem.set_data_handle(grad_bias_data);
    args[DNNL_ARG_DIFF_BIAS] = handle->bwd_weights_diff_bias_mem;
  }

  if (handle->has_bwd_weights_scratchpad) {
    if (!workspace_data) {
      throw std::runtime_error(
          "dnnl_conv2d run_backward_weights_and_bias: scratchpad required but workspace is null");
    }
    handle->bwd_weights_scratchpad_mem.set_data_handle(workspace_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->bwd_weights_scratchpad_mem;
  }

  handle->bwd_weights_conv.execute(s, args);
  if (handle->needs_weights_reorder) {
    handle->user_grad_weights_mem.set_data_handle(grad_weight_data);
    handle->grad_weights_to_user_reorder.execute(s, {{DNNL_ARG_FROM, handle->packed_grad_weights},
                                                     {DNNL_ARG_TO, handle->user_grad_weights_mem}});
  }
  s.wait();
}

}  // namespace dnnl_conv2d
}  // namespace cpu
}  // namespace tnn

#endif
