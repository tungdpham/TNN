/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/dnnl_batchnorm_ops.hpp"

#ifdef USE_DNNL

#include <dnnl.hpp>
#include <stdexcept>
#include <unordered_map>

#include "nn/layers_impl/common/batchnorm.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dnnl_batchnorm {

struct dnnlBNHandle_t {
  dnnl::engine engine;
  dnnl::stream stream;

  dnnl::batch_normalization_forward fwd_prim;   // forward_training
  dnnl::batch_normalization_forward inf_prim;   // forward_inference
  dnnl::batch_normalization_backward bwd_prim;  // backward (or run_dgrad)

  // src / dst — NHWC, io dtype
  dnnl::memory fwd_src_mem, fwd_dst_mem;
  dnnl::memory inf_src_mem, inf_dst_mem;
  dnnl::memory bwd_src_mem, bwd_diff_dst_mem, bwd_diff_src_mem;

  // scale / shift (gamma/beta) — FP32 {C}
  dnnl::memory fwd_scale_mem, fwd_shift_mem;
  dnnl::memory inf_scale_mem, inf_shift_mem;
  dnnl::memory bwd_scale_mem, bwd_diff_scale_mem, bwd_diff_shift_mem;

  // mean / variance — FP32 {C}, output of fwd_training, input of bwd
  dnnl::memory fwd_mean_mem, fwd_var_mem;
  dnnl::memory bwd_mean_mem, bwd_var_mem;

  // ReLU workspace — only valid when has_relu
  dnnl::memory fwd_workspace_mem;
  dnnl::memory bwd_workspace_mem;

  // Scratchpads — null handle by default
  dnnl::memory fwd_scratchpad_mem;
  dnnl::memory inf_scratchpad_mem;
  dnnl::memory bwd_scratchpad_mem;

  bool has_relu = false;
  bool has_affine = false;
  bool has_affine_bwd = false;  // true when prop_kind::backward (computes diff_scale/shift)
};

static dnnl::memory::data_type get_dnnl_dtype(DType_t dtype) {
  switch (dtype) {
    case DType_t::FP32:
      return dnnl::memory::data_type::f32;
    case DType_t::FP16:
      return dnnl::memory::data_type::f16;
    case DType_t::BF16:
      return dnnl::memory::data_type::bf16;
    default:
      throw std::runtime_error("dnnl_batchnorm: unsupported dtype");
  }
}

dnnlBNHandle_t *initialize_dnnl_handle(BatchNormStats &stats, DType_t dtype) {
  auto *handle = new dnnlBNHandle_t();
  handle->engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  handle->stream = dnnl::stream(handle->engine);

  handle->has_relu = stats.use_relu;
  handle->has_affine = stats.affine;
  handle->has_affine_bwd = stats.affine;

  auto io_dt = get_dnnl_dtype(dtype);
  const auto f32_dt = dnnl::memory::data_type::f32;

  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);
  const int64_t h = static_cast<int64_t>(stats.height);
  const int64_t w = static_cast<int64_t>(stats.width);

  // NHWC layout
  auto src_md = dnnl::memory::desc({n, c, h, w}, io_dt, dnnl::memory::format_tag::nhwc);
  auto scale_md = dnnl::memory::desc({c}, f32_dt, dnnl::memory::format_tag::a);
  auto shift_md = dnnl::memory::desc({c}, f32_dt, dnnl::memory::format_tag::a);

  dnnl::normalization_flags flags = dnnl::normalization_flags::none;
  if (stats.affine) {
    flags |= dnnl::normalization_flags::use_scale;
    flags |= dnnl::normalization_flags::use_shift;
  }
  if (stats.use_relu) {
    flags |= dnnl::normalization_flags::fuse_norm_relu;
  }

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  dnnl::batch_normalization_forward::primitive_desc fwd_pd;

  // Forward training
  {
    fwd_pd = dnnl::batch_normalization_forward::primitive_desc(
        handle->engine, dnnl::prop_kind::forward_training, src_md, src_md,
        static_cast<float>(stats.epsilon), flags, attr);

    handle->fwd_prim = dnnl::batch_normalization_forward(fwd_pd);
    stats.fwd_workspace_size = fwd_pd.scratchpad_desc().get_size();
    stats.relu_workspace_size = stats.use_relu ? fwd_pd.workspace_desc().get_size() : 0;

    handle->fwd_src_mem = dnnl::memory(src_md, handle->engine, nullptr);
    handle->fwd_dst_mem = dnnl::memory(src_md, handle->engine, nullptr);
    handle->fwd_mean_mem = dnnl::memory(fwd_pd.mean_desc(), handle->engine, nullptr);
    handle->fwd_var_mem = dnnl::memory(fwd_pd.variance_desc(), handle->engine, nullptr);

    if (stats.affine) {
      handle->fwd_scale_mem = dnnl::memory(scale_md, handle->engine, nullptr);
      handle->fwd_shift_mem = dnnl::memory(shift_md, handle->engine, nullptr);
    }
    if (stats.fwd_workspace_size > 0) {
      handle->fwd_scratchpad_mem = dnnl::memory(fwd_pd.scratchpad_desc(), handle->engine, nullptr);
    }
    if (stats.use_relu) {
      handle->fwd_workspace_mem = dnnl::memory(fwd_pd.workspace_desc(), handle->engine, nullptr);
    }
  }

  // Forward inference
  {
    auto inf_pd = dnnl::batch_normalization_forward::primitive_desc(
        handle->engine, dnnl::prop_kind::forward_inference, src_md, src_md,
        static_cast<float>(stats.epsilon), flags, attr);

    handle->inf_prim = dnnl::batch_normalization_forward(inf_pd);
    stats.inf_workspace_size = inf_pd.scratchpad_desc().get_size();

    handle->inf_src_mem = dnnl::memory(src_md, handle->engine, nullptr);
    handle->inf_dst_mem = dnnl::memory(src_md, handle->engine, nullptr);

    if (stats.affine) {
      handle->inf_scale_mem = dnnl::memory(scale_md, handle->engine, nullptr);
      handle->inf_shift_mem = dnnl::memory(shift_md, handle->engine, nullptr);
    }
    if (stats.inf_workspace_size > 0) {
      handle->inf_scratchpad_mem = dnnl::memory(inf_pd.scratchpad_desc(), handle->engine, nullptr);
    }
  }

  // Backward
  {
    // Use backward when affine (computes diff_scale/shift), otherwise backward_data.
    dnnl::prop_kind bwd_pk =
        stats.affine ? dnnl::prop_kind::backward : dnnl::prop_kind::backward_data;

    auto bwd_pd = dnnl::batch_normalization_backward::primitive_desc(
        handle->engine, bwd_pk, src_md, src_md, src_md, static_cast<float>(stats.epsilon), flags,
        fwd_pd, attr);

    handle->bwd_prim = dnnl::batch_normalization_backward(bwd_pd);
    stats.bwd_workspace_size = bwd_pd.scratchpad_desc().get_size();

    handle->bwd_src_mem = dnnl::memory(src_md, handle->engine, nullptr);
    handle->bwd_diff_dst_mem = dnnl::memory(src_md, handle->engine, nullptr);
    handle->bwd_diff_src_mem = dnnl::memory(src_md, handle->engine, nullptr);
    handle->bwd_mean_mem = dnnl::memory(bwd_pd.mean_desc(), handle->engine, nullptr);
    handle->bwd_var_mem = dnnl::memory(bwd_pd.variance_desc(), handle->engine, nullptr);

    if (stats.affine) {
      handle->bwd_scale_mem = dnnl::memory(scale_md, handle->engine, nullptr);
      // diff_scale and diff_shift share the same {C} f32 layout as scale and shift
      handle->bwd_diff_scale_mem = dnnl::memory(scale_md, handle->engine, nullptr);
      handle->bwd_diff_shift_mem = dnnl::memory(shift_md, handle->engine, nullptr);
    }
    if (stats.bwd_workspace_size > 0) {
      handle->bwd_scratchpad_mem = dnnl::memory(bwd_pd.scratchpad_desc(), handle->engine, nullptr);
    }
    if (stats.use_relu) {
      handle->bwd_workspace_mem = dnnl::memory(bwd_pd.workspace_desc(), handle->engine, nullptr);
    }
  }

  round_workspace_size(stats);
  return handle;
}

void destroy_dnnl_handle(dnnlBNHandle_t *handle) { delete handle; }

void run_forward(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                 const void *scale_data, const void *shift_data, void *output_data, void *mean_data,
                 void *var_data, void *relu_ws_data, void *scratchpad_data) {
  dnnl::stream &s = handle->stream;

  handle->fwd_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->fwd_dst_mem.set_data_handle(output_data);
  handle->fwd_mean_mem.set_data_handle(mean_data);
  handle->fwd_var_mem.set_data_handle(var_data);

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, handle->fwd_src_mem},
      {DNNL_ARG_DST, handle->fwd_dst_mem},
      {DNNL_ARG_MEAN, handle->fwd_mean_mem},
      {DNNL_ARG_VARIANCE, handle->fwd_var_mem},
  };

  if (handle->has_affine) {
    handle->fwd_scale_mem.set_data_handle(const_cast<void *>(scale_data));
    handle->fwd_shift_mem.set_data_handle(const_cast<void *>(shift_data));
    args[DNNL_ARG_SCALE] = handle->fwd_scale_mem;
    args[DNNL_ARG_SHIFT] = handle->fwd_shift_mem;
  }

  if (handle->fwd_scratchpad_mem) {
    if (!scratchpad_data) throw std::runtime_error("dnnl_batchnorm run_forward: null scratchpad");
    handle->fwd_scratchpad_mem.set_data_handle(scratchpad_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->fwd_scratchpad_mem;
  }

  if (handle->has_relu) {
    if (!relu_ws_data) throw std::runtime_error("dnnl_batchnorm run_forward: null relu workspace");
    handle->fwd_workspace_mem.set_data_handle(relu_ws_data);
    args[DNNL_ARG_WORKSPACE] = handle->fwd_workspace_mem;
  }

  // auto start_time = std::chrono::high_resolution_clock::now();
  handle->fwd_prim.execute(s, args);
  s.wait();

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration_ms =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "DNNL batchnorm forward training pass took " << duration_ms << " ms" << std::endl;
}

void run_inference(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                   const void *scale_data, const void *shift_data, const void *mean_data,
                   const void *var_data, void *output_data, void *scratchpad_data) {
  dnnl::stream &s = handle->stream;

  handle->inf_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->inf_dst_mem.set_data_handle(output_data);

  // Inference reads mean/var from separate memory objects — reuse bwd_mean/var which hold f32 {C}
  // For inference we use the fwd_mean/fwd_var mem objects as input containers.
  // Actually re-use the training mean/var memory objects (same descriptor) for inference.
  handle->fwd_mean_mem.set_data_handle(const_cast<void *>(mean_data));
  handle->fwd_var_mem.set_data_handle(const_cast<void *>(var_data));

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, handle->inf_src_mem},
      {DNNL_ARG_DST, handle->inf_dst_mem},
      {DNNL_ARG_MEAN, handle->fwd_mean_mem},
      {DNNL_ARG_VARIANCE, handle->fwd_var_mem},
  };

  if (handle->has_affine) {
    handle->inf_scale_mem.set_data_handle(const_cast<void *>(scale_data));
    handle->inf_shift_mem.set_data_handle(const_cast<void *>(shift_data));
    args[DNNL_ARG_SCALE] = handle->inf_scale_mem;
    args[DNNL_ARG_SHIFT] = handle->inf_shift_mem;
  }

  if (handle->inf_scratchpad_mem) {
    if (!scratchpad_data) throw std::runtime_error("dnnl_batchnorm run_inference: null scratchpad");
    handle->inf_scratchpad_mem.set_data_handle(scratchpad_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->inf_scratchpad_mem;
  }

  handle->inf_prim.execute(s, args);
  s.wait();
}

void run_backward(dnnlBNHandle_t *handle, const BatchNormStats &stats, const void *input_data,
                  const void *grad_output_data, void *grad_input_data, const void *scale_data,
                  void *d_scale_data, void *d_shift_data, const void *mean_data,
                  const void *var_data, const void *relu_ws_data, void *scratchpad_data) {
  dnnl::stream &s = handle->stream;

  handle->bwd_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->bwd_diff_dst_mem.set_data_handle(const_cast<void *>(grad_output_data));
  handle->bwd_diff_src_mem.set_data_handle(grad_input_data);
  handle->bwd_mean_mem.set_data_handle(const_cast<void *>(mean_data));
  handle->bwd_var_mem.set_data_handle(const_cast<void *>(var_data));

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_SRC, handle->bwd_src_mem},           {DNNL_ARG_DIFF_DST, handle->bwd_diff_dst_mem},
      {DNNL_ARG_DIFF_SRC, handle->bwd_diff_src_mem}, {DNNL_ARG_MEAN, handle->bwd_mean_mem},
      {DNNL_ARG_VARIANCE, handle->bwd_var_mem},
  };

  if (handle->has_affine_bwd) {
    handle->bwd_scale_mem.set_data_handle(const_cast<void *>(scale_data));
    handle->bwd_diff_scale_mem.set_data_handle(d_scale_data);
    handle->bwd_diff_shift_mem.set_data_handle(d_shift_data);
    args[DNNL_ARG_SCALE] = handle->bwd_scale_mem;
    args[DNNL_ARG_DIFF_SCALE] = handle->bwd_diff_scale_mem;
    args[DNNL_ARG_DIFF_SHIFT] = handle->bwd_diff_shift_mem;
  }

  if (handle->bwd_scratchpad_mem) {
    if (!scratchpad_data) throw std::runtime_error("dnnl_batchnorm run_backward: null scratchpad");
    handle->bwd_scratchpad_mem.set_data_handle(scratchpad_data);
    args[DNNL_ARG_SCRATCHPAD] = handle->bwd_scratchpad_mem;
  }

  if (handle->has_relu) {
    if (!relu_ws_data) throw std::runtime_error("dnnl_batchnorm run_backward: null relu workspace");
    handle->bwd_workspace_mem.set_data_handle(const_cast<void *>(relu_ws_data));
    args[DNNL_ARG_WORKSPACE] = handle->bwd_workspace_mem;
  }

  // auto start_time = std::chrono::high_resolution_clock::now();
  handle->bwd_prim.execute(s, args);
  s.wait();
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration_ms =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  // std::cout << "DNNL batchnorm backward pass took " << duration_ms << " ms" << std::endl;
}

}  // namespace dnnl_batchnorm
}  // namespace cpu
}  // namespace tnn

#endif  // USE_DNNL
