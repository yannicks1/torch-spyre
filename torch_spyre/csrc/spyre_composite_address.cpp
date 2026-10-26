/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_composite_address.h"

#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "spyre_allocator.h"
#include "spyre_tensor_impl.h"

namespace spyre {

namespace {

// Storage-level safety checks: on-device, non-null storage and context.
// Contiguity is NOT checked here -- a non-contiguous view of a Spyre tensor
// still shares its storage's CompositeAddress with the base allocation, and
// C++ callers (D2H copies of transposed views, etc.) legitimately need it.
// The Python handle path layers a contiguity check on top for callers that
// want a well-defined chunk geometry.
SharedOwnerCtx* resolve_owner_ctx(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone(),
              "get_composite_address: tensor must be on the Spyre device");

  const auto& storage = tensor.storage();
  TORCH_CHECK(storage.data_ptr().get() != nullptr,
              "get_composite_address: storage data pointer is null");

  auto* ctx = static_cast<SharedOwnerCtx*>(storage.data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr,
              "get_composite_address: SharedOwnerCtx is null (tensor has no "
              "Spyre device allocation)");
  return ctx;
}

}  // namespace

CompositeAddressHandle::CompositeAddressHandle(const at::Tensor& tensor)
    // Keepalive: retain the tensor's storage so the SharedOwnerCtx below (and
    // the CompositeAddress it owns) cannot be freed while this handle lives.
    : storage_(tensor.storage()), composite_addr_([&]() {
        TORCH_CHECK(tensor.is_contiguous(),
                    "get_composite_address: tensor must be contiguous");
        return &resolve_owner_ctx(tensor)->composite_addr;
      }()) {}

flex::CompositeAddress* get_composite_address(const at::Tensor& tensor) {
  return &resolve_owner_ctx(tensor)->composite_addr;
}

size_t CompositeAddressHandle::total_size() const {
  return composite_addr_->total_size();
}

size_t CompositeAddressHandle::num_chunks() const {
  return composite_addr_->chunks().size();
}

std::vector<CompositeChunkInfo> CompositeAddressHandle::chunks() const {
  const auto& chunks = composite_addr_->chunks();
  std::vector<CompositeChunkInfo> out;
  out.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    out.push_back(CompositeChunkInfo{
        /*region_id=*/chunk.addr.region_id,
        /*offset=*/chunk.addr.offset,
        /*size=*/chunk.size,
        /*domain_id=*/chunk.domain_id,
    });
  }
  return out;
}

CompositeAddressHandle get_composite_address_handle(const at::Tensor& tensor) {
  return CompositeAddressHandle(tensor);
}

namespace {

// Device coordinate of a host element offset, peeled from the largest host
// stride down as the D2H copy does (spyre_mem.cpp). Empty if it does not
// decompose exactly.
std::vector<int64_t> device_coordinate(const SpyreTensorLayout& stl,
                                       int64_t host_offset) {
  const int rank = static_cast<int>(stl.stride_map.size());
  std::map<int64_t, int, std::greater<int64_t>> stride_to_dim;
  for (int i = 0; i < rank; i++) {
    if (stl.stride_map[i] <= 0 || stl.device_size[i] <= 1) continue;
    stride_to_dim.emplace(stl.stride_map[i], i);
  }
  std::vector<int64_t> coord(rank, 0);
  for (const auto& [stride, dim] : stride_to_dim) {
    if (stride > host_offset) continue;
    coord[dim] = host_offset / stride;
    host_offset -= coord[dim] * stride;
  }
  if (host_offset != 0) return {};
  return coord;
}

}  // namespace

int64_t rebase_offset_bytes(const at::Tensor& tensor) {
  if (!tensor.is_privateuseone()) return -1;
  const int64_t host_offset = tensor.storage_offset();
  if (host_offset == 0) return 0;
  auto* impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr || tensor.numel() == 0) return -1;
  const SpyreTensorLayout& stl = impl->spyre_layout;
  const int rank = static_cast<int>(stl.stride_map.size());
  const int64_t eps = stl.elems_per_stick();
  if (rank == 0 || eps <= 0) return -1;

  // The kernel reads element h at device position f(h), so the rebased view
  // reads f(h) from base + f(offset). f is linear in the device coordinate, so
  // that is f(offset + h) when adding the offset carries into no device dim;
  // the view's last element has the largest digit in every dim. The offset
  // must also leave the stick dim (the last one) alone.
  int64_t last = 0;
  for (int64_t d = 0; d < tensor.dim(); d++) {
    last += (tensor.size(d) - 1) * tensor.stride(d);
  }
  const auto c_off = device_coordinate(stl, host_offset);
  const auto c_last = device_coordinate(stl, last);
  const auto c_end = device_coordinate(stl, host_offset + last);
  if (c_off.empty() || c_last.empty() || c_end.empty()) return -1;
  if (stl.device_size[rank - 1] != eps || c_off[rank - 1] != 0) return -1;
  for (int i = 0; i < rank; i++) {
    if (c_off[i] + c_last[i] != c_end[i]) return -1;
    if (c_end[i] >= std::max<int64_t>(stl.device_size[i], 1)) return -1;
  }

  int64_t device_offset = 0;
  int64_t device_stride = 1;
  for (int i = rank - 1; i >= 0; i--) {
    device_offset += c_off[i] * device_stride;
    device_stride *= stl.device_size[i];
  }

  const auto& addr = resolve_owner_ctx(tensor)->composite_addr;
  if (addr.chunks().size() != 1) return -1;
  const int64_t stick_bytes = static_cast<int64_t>(
      get_device_size_in_bytes(std::vector<int64_t>{eps}, stl.device_dtype));
  const int64_t byte_offset = device_offset / eps * stick_bytes;
  if (byte_offset >= static_cast<int64_t>(addr.total_size())) return -1;
  return byte_offset;
}

bool is_rebasable_offset_view(const at::Tensor& tensor) {
  return tensor.is_privateuseone() && tensor.storage_offset() != 0 &&
         rebase_offset_bytes(tensor) > 0;
}

namespace {

// Owner context of a rebased storage: a non-owning address into the original
// allocation, plus a reference that keeps that allocation alive.
struct RebasedOwnerCtx : SharedOwnerCtx {
  c10::Storage keepalive;

  RebasedOwnerCtx(flex::CompositeAddress addr, signed char dev_id,
                  flex::MemoryType mem_type, c10::Storage base)
      : SharedOwnerCtx(std::move(addr), dev_id, mem_type),
        keepalive(std::move(base)) {}
};

void delete_rebased_ctx(void* ctx) {
  delete static_cast<RebasedOwnerCtx*>(static_cast<SharedOwnerCtx*>(ctx));
}

}  // namespace

at::Tensor rebase_offset_view(const at::Tensor& tensor) {
  const int64_t byte_offset = rebase_offset_bytes(tensor);
  if (byte_offset <= 0) return tensor;
  const SharedOwnerCtx* base_ctx = resolve_owner_ctx(tensor);
  const auto& chunk = base_ctx->composite_addr.chunks()[0];
  const size_t nbytes = chunk.size - byte_offset;
  flex::CompositeAddress shifted(
      flex::Chunk(flex::LogicalAddress(chunk.addr.region_id,
                                       chunk.addr.offset + byte_offset),
                  nbytes, chunk.domain_id));

  // resolve_owner_ctx reads the context back as a SharedOwnerCtx*.
  SharedOwnerCtx* ctx =
      new RebasedOwnerCtx(std::move(shifted), base_ctx->device_id,
                          base_ctx->memory_type, tensor.storage());
  c10::DataPtr data_ptr(ctx, ctx, &delete_rebased_ctx, tensor.device());
  auto storage = c10::Storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes, std::move(data_ptr),
      /*allocator=*/nullptr, /*resizable=*/false));

  auto* orig = static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  at::Tensor result = at::detail::make_tensor<SpyreTensorImpl>(
      std::move(storage), tensor.key_set(), tensor.dtype());
  auto* impl = static_cast<SpyreTensorImpl*>(result.unsafeGetTensorImpl());
  impl->set_sizes_and_strides(tensor.sizes(), tensor.strides());
  impl->spyre_layout = orig->spyre_layout;
  impl->dma_sizes = orig->dma_sizes;
  impl->dma_strides = orig->dma_strides;
  return result;
}

}  // namespace spyre
