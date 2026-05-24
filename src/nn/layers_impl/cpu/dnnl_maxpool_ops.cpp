/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/layers_impl/cpu/dnnl_maxpool_ops.hpp"

#ifdef USE_DNNL

#include <dnnl.hpp>
#include <stdexcept>

#include "nn/layers_impl/common/maxpool.hpp"
#include "type/type.hpp"

namespace tnn {
namespace cpu {
namespace dnnl_maxpool {

struct dnnlMaxPoolHandle_t {
  dnnl::engine engine;
  dnnl::stream stream;

  dnnl::pooling_forward fwd_pool;
  dnnl::pooling_forward inf_pool;
  dnnl::pooling_backward bwd_pool;

  dnnl::memory fwd_user_src_mem, fwd_user_dst_mem;
  dnnl::memory::desc pool_workspace_md;

  dnnl::memory inf_user_src_mem, inf_user_dst_mem;

  dnnl::memory bwd_user_dst_mem, bwd_user_src_mem;
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
      throw std::runtime_error("dnnl_maxpool: unsupported dtype");
  }
}

dnnlMaxPoolHandle_t *initialize_dnnl_handle(MaxPoolStats &stats, DType_t dtype) {
  auto *handle = new dnnlMaxPoolHandle_t();
  handle->engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
  handle->stream = dnnl::stream(handle->engine);

  auto dt = get_dnnl_dtype(dtype);

  const int64_t n = static_cast<int64_t>(stats.batch_size);
  const int64_t c = static_cast<int64_t>(stats.channels);
  const int64_t ih = static_cast<int64_t>(stats.input_h);
  const int64_t iw = static_cast<int64_t>(stats.input_w);
  const int64_t oh = static_cast<int64_t>(stats.output_h);
  const int64_t ow = static_cast<int64_t>(stats.output_w);
  const int64_t kh = static_cast<int64_t>(stats.pool_h);
  const int64_t kw = static_cast<int64_t>(stats.pool_w);
  const int64_t sh = static_cast<int64_t>(stats.stride_h);
  const int64_t sw = static_cast<int64_t>(stats.stride_w);
  const int64_t ph = static_cast<int64_t>(stats.pad_h);
  const int64_t pw = static_cast<int64_t>(stats.pad_w);

  auto user_src_md = dnnl::memory::desc({n, c, ih, iw}, dt, dnnl::memory::format_tag::nhwc);
  auto user_dst_md = dnnl::memory::desc({n, c, oh, ow}, dt, dnnl::memory::format_tag::nhwc);

  dnnl::memory::dims strides = {sh, sw};
  dnnl::memory::dims kernel = {kh, kw};
  dnnl::memory::dims dilation = {0, 0};
  dnnl::memory::dims pad_l = {ph, pw};
  dnnl::memory::dims pad_r = {ph, pw};

  auto fwd_pd = dnnl::pooling_forward::primitive_desc(
      handle->engine, dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max, user_src_md,
      user_dst_md, strides, kernel, dilation, pad_l, pad_r);

  handle->fwd_pool = dnnl::pooling_forward(fwd_pd);
  stats.fwd_workspace_size = 0;
  stats.pool_workspace_size = static_cast<size_t>(fwd_pd.workspace_desc().get_size());
  handle->pool_workspace_md = fwd_pd.workspace_desc();

  handle->fwd_user_src_mem = dnnl::memory(user_src_md, handle->engine, nullptr);
  handle->fwd_user_dst_mem = dnnl::memory(user_dst_md, handle->engine, nullptr);

  {
    auto inf_pd = dnnl::pooling_forward::primitive_desc(
        handle->engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
        user_src_md, user_dst_md, strides, kernel, dilation, pad_l, pad_r);

    handle->inf_pool = dnnl::pooling_forward(inf_pd);
    stats.inf_workspace_size = 0;

    handle->inf_user_src_mem = dnnl::memory(user_src_md, handle->engine, nullptr);
    handle->inf_user_dst_mem = dnnl::memory(user_dst_md, handle->engine, nullptr);
  }

  {
    auto bwd_pd = dnnl::pooling_backward::primitive_desc(
        handle->engine, dnnl::algorithm::pooling_max, user_src_md, user_dst_md, strides, kernel,
        dilation, pad_l, pad_r, fwd_pd);

    handle->bwd_pool = dnnl::pooling_backward(bwd_pd);
    stats.bwd_workspace_size = 0;

    handle->bwd_user_dst_mem = dnnl::memory(user_dst_md, handle->engine, nullptr);
    handle->bwd_user_src_mem = dnnl::memory(user_src_md, handle->engine, nullptr);
  }

  round_workspace_size(stats);
  return handle;
}

void destroy_dnnl_handle(dnnlMaxPoolHandle_t *handle) { delete handle; }

void run_forward(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats, const void *input_data,
                 void *output_data, void *pool_workspace_data, void * /*scratchpad_data*/) {
  dnnl::stream &s = handle->stream;

  handle->fwd_user_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->fwd_user_dst_mem.set_data_handle(output_data);

  if (stats.pool_workspace_size == 0) {
    throw std::runtime_error("dnnl_maxpool run_forward: pool_workspace_size is 0 for pooling_max");
  }
  dnnl::memory pool_ws_mem(handle->pool_workspace_md, handle->engine, pool_workspace_data);

  // auto start = std::chrono::high_resolution_clock::now();
  handle->fwd_pool.execute(s, {{DNNL_ARG_SRC, handle->fwd_user_src_mem},
                               {DNNL_ARG_DST, handle->fwd_user_dst_mem},
                               {DNNL_ARG_WORKSPACE, pool_ws_mem}});
  s.wait();
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed = end - start;
  // std::cout << "DNNL MaxPool forward execution time: " << elapsed.count() << " seconds"
  //           << std::endl;
}

void run_inference(dnnlMaxPoolHandle_t *handle, const MaxPoolStats & /*stats*/,
                   const void *input_data, void *output_data, void * /*scratchpad_data*/) {
  dnnl::stream &s = handle->stream;

  handle->inf_user_src_mem.set_data_handle(const_cast<void *>(input_data));
  handle->inf_user_dst_mem.set_data_handle(output_data);

  handle->inf_pool.execute(
      s, {{DNNL_ARG_SRC, handle->inf_user_src_mem}, {DNNL_ARG_DST, handle->inf_user_dst_mem}});
  s.wait();
}

void run_backward(dnnlMaxPoolHandle_t *handle, const MaxPoolStats &stats,
                  const void *grad_output_data, void *grad_input_data,
                  const void *pool_workspace_data, void * /*scratchpad_data*/) {
  dnnl::stream &s = handle->stream;

  handle->bwd_user_dst_mem.set_data_handle(const_cast<void *>(grad_output_data));
  handle->bwd_user_src_mem.set_data_handle(grad_input_data);

  dnnl::memory pool_ws_mem(handle->pool_workspace_md, handle->engine,
                           const_cast<void *>(pool_workspace_data));

  // auto start = std::chrono::high_resolution_clock::now();
  handle->bwd_pool.execute(s, {{DNNL_ARG_DIFF_DST, handle->bwd_user_dst_mem},
                               {DNNL_ARG_DIFF_SRC, handle->bwd_user_src_mem},
                               {DNNL_ARG_WORKSPACE, pool_ws_mem}});
  s.wait();
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed = end - start;
  // std::cout << "DNNL MaxPool backward execution time: " << elapsed.count() << " seconds"
  //           << std::endl;
}

}  // namespace dnnl_maxpool
}  // namespace cpu
}  // namespace tnn

#endif
