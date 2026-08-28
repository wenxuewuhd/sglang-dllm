from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKOnlyPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu


def _init_npu_conv_state(
    conv_state_in,
    conv_state_shape,
    speculative_num_draft_tokens: Optional[int] = None,
    is_kda: bool = False,
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # Mamba shapes are (channels, window), while KDA shapes are
    # (window, channels). NPU kernels consume KDA state as
    # [layers, pool, channels, window] and other Mamba state as
    # [layers, pool, window, channels]. KDA keeps the base window fixed;
    # speculative per-step windows live in the intermediate cache.
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                conv_shape[1] if is_kda else conv_shape[1] + extra_conv_len,
                conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        **kwargs,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.use_triton_prefix_kv_cache_store = (
            envs.SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE.get()
        )
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
            **kwargs,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.v_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )

            if self.use_fia:
                # Use per-layer Python lists to avoid torch.compile capturing
                # the entire multi-layer tensor (OOM during graph capture).
                # Each layer view: [P*ps, 1, H, D], sharing the contiguous
                # storage allocated above.
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]
                self.v_buffer = [
                    self.v_buffer[i].view(-1, 1, self.head_num, self.v_head_dim)
                    for i in range(self.layer_num)
                ]

    def _init_kv_copy_and_warmup(self):
        # implementation relies on self.data_strides / self.data_ptrs, which the
        # NPU paged buffer layout never builds.
        self._kv_copy_config = None

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
            num_rows = loc.numel()
            expected_k_numel = num_rows * self.head_num * self.head_dim
            expected_v_numel = num_rows * self.head_num * self.v_head_dim
            if (
                cache_k.numel() != expected_k_numel
                or cache_v.numel() != expected_v_numel
            ):
                raise ValueError(
                    "NPU FIA KV scatter row mismatch: "
                    f"loc_rows={num_rows}, cache_k_shape={tuple(cache_k.shape)}, "
                    f"cache_v_shape={tuple(cache_v.shape)}, "
                    f"head_num={self.head_num}, head_dim={self.head_dim}, "
                    f"v_head_dim={self.v_head_dim}."
                )

            # aclnnScatterNdUpdate on the deployed CANN rejects the otherwise
            # valid 4-D [slot, 1, head, dim] update during tiling. Flatten only
            # the singleton FIA layout axis and scatter through an equivalent
            # 3-D view; the underlying KV storage and attention layout stay
            # unchanged.
            loc_indices = loc.contiguous().view(-1, 1)
            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer.view(-1, self.head_num, self.head_dim),
                loc_indices,
                cache_k.contiguous().view(num_rows, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
                loc_indices,
                cache_v.contiguous().view(num_rows, self.head_num, self.v_head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.v_head_dim
                ),
                slot_indices=loc,
            )

    def set_kv_buffer_prefix_valid(
        self,
        layer: "RadixAttention",
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if not self.use_triton_prefix_kv_cache_store:
            return super().set_kv_buffer_prefix_valid(
                layer,
                loc_2d,
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override,
            )

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got {tuple(loc_2d.shape)}")

        num_rows = loc_2d.numel()
        if (
            cache_k.numel() != num_rows * self.head_num * self.head_dim
            or cache_v.numel() != num_rows * self.head_num * self.v_head_dim
        ):
            raise ValueError(
                "dense NPU KV rows must match loc_2d size: "
                f"cache_k={tuple(cache_k.shape)}, cache_v={tuple(cache_v.shape)}, "
                f"loc_2d={tuple(loc_2d.shape)}"
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
        v_buffer_layer = self.v_buffer[layer_id - self.start_layer]
        if loc_2d.device != k_buffer_layer.device:
            loc_2d = loc_2d.to(device=k_buffer_layer.device, non_blocking=True)
        if commit_lens.device != k_buffer_layer.device:
            commit_lens = commit_lens.to(
                device=k_buffer_layer.device, non_blocking=True
            )
        self._debug_prefix_valid_backend = "npu_triton"
        from sgl_kernel_npu.mem_cache.kv_cache_store import (
            store_kv_cache_prefix_valid_npu_triton,
        )

        store_kv_cache_prefix_valid_npu_triton(
            k_buffer_layer.view(-1, self.head_num, self.head_dim),
            v_buffer_layer.view(-1, self.head_num, self.v_head_dim),
            cache_k.reshape(num_rows, self.head_num, self.head_dim),
            cache_v.reshape(num_rows, self.head_num, self.v_head_dim),
            loc_2d,
            commit_lens,
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMHATokenToKOnlyPool(MHATokenToKOnlyPool):
    """NPU paged K-only cache used by MiniMax sparse index-only layers."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super(MHATokenToKOnlyPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = torch.zeros(
                (
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            if self.use_fia:
                self.k_buffer = [
                    self.k_buffer[i].view(-1, 1, self.head_num, self.head_dim)
                    for i in range(self.layer_num)
                ]

        self._finalize_allocation_log(size)

    def _get_key_buffer(self, layer_id: int):
        k_buffer = self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return k_buffer.view(self.dtype)
        return k_buffer

    def set_k_buffer(
        self,
        layer_id: int,
        loc_info,
        cache_k: torch.Tensor,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)

        k_buffer_layer = self.k_buffer[layer_id - self.start_layer].view(
            -1, self.head_num, self.head_dim
        )
        loc = loc.to(device=cache_k.device, dtype=torch.int32).contiguous()
        torch_npu.npu_scatter_nd_update_(
            k_buffer_layer,
            loc.view(-1, 1),
            cache_k.contiguous().view(-1, self.head_num, self.head_dim),
        )

    def get_contiguous_buf_infos(self):
        data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        return get_tensor_size_bytes(self.k_buffer), 0


class NPUMiniMaxSparseKVPool(MiniMaxSparseKVPool):
    """MiniMax sparse wrapper backed by NPU paged MHA/index pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            main_pool_cls=NPUMHATokenToKVPool,
            index_kv_pool_cls=NPUMHATokenToKVPool,
            index_k_pool_cls=NPUMHATokenToKOnlyPool,
            **kwargs,
        )

    def get_index_k_state_buf_infos(self):
        pool = self.index_k_pool
        n = pool.layer_num
        data_ptrs = [pool.get_key_buffer(i).data_ptr() for i in range(n)]
        data_lens = [pool.get_key_buffer(i).nbytes for i in range(n)]
        if pool.use_fia:
            item_lens = [
                pool.get_key_buffer(i)[0].nbytes * pool.page_size for i in range(n)
            ]
        else:
            item_lens = [pool.get_key_buffer(i)[0].nbytes for i in range(n)]
        return data_ptrs, data_lens, item_lens


class NPUMLATokenToKVPool(MLATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        index_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )

        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        data_ptrs = [self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)]
        data_lens = [self.index_k_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.index_k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.index_k_buffer[layer_id - self.start_layer]

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        torch_npu.npu_scatter_nd_update_(
            self.k_buffer[layer_id - self.start_layer].view(-1, 1, self.kv_lora_rank),
            loc.view(-1, 1),
            cache_k.view(-1, 1, self.kv_lora_rank),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_buffer[layer_id - self.start_layer].view(
                -1, 1, self.qk_rope_head_dim
            ),
            loc.view(-1, 1),
            cache_v.view(-1, 1, self.qk_rope_head_dim),
        )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id - self.start_layer].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = []
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            ik_layer = (
                self.index_k_buffer[local_layer_id].view(-1, 1, self.index_head_dim)
                if has_ik
                else None
            )
            buf_of_layers.append([k_layer, v_layer, ik_layer])

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            ik_layer = (
                self.index_k_buffer[local_layer_id].view(-1, 1, self.index_head_dim)
                if has_ik
                else None
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                assert k_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
                if has_ik:
                    ik_cpu = chunk[2]
                    ik_layer[chunk_indices] = ik_cpu.to(
                        ik_layer.device, non_blocking=True
                    )
        torch.npu.synchronize()


class NPUBf16IndexKeyCache(IndexKeyCache):
    """kpool's compressed index keys as bf16, laid out the way the operator reads.

    The shared cache packs an fp8 key and an fp32 scale into one uint8 page.
    Atlas A3 cannot hold an fp8 tensor at all -- allocating one raises
    ``aclnnInplaceZero 161002`` -- and the operator that scores these keys,
    ``torch_npu.npu_lightning_indexer``, reads bf16 and nothing else. So the
    scale region goes away and the page becomes ``PA_BSND``:
    ``(pages, page_size, 1, index_head_dim)``, which is the shape the operator
    wants with no restride at the call site.
    """

    def _buffer_shape(self, num_pages: int) -> tuple[int, ...]:
        # One page beyond what the block table can address, reserved as a place
        # for masked-off rows to write. See NPUDSATokenToKVPool.scratch_loc.
        pool = self.pool
        return (num_pages + 1, pool.page_size, 1, pool.index_head_dim)


class NPUDSATokenToKVPool(DSATokenToKVPool):
    """DSA pool whose index-K cache is bf16 rather than packed fp8.

    Everything else -- the latent KV, the bf16 compress-tail ring, the page
    bookkeeping -- is the shared implementation. Only the index cache and the two
    writers that quantize into it change.
    """

    index_k_with_scale_buffer_dtype = torch.bfloat16

    def _create_index_key_cache(self) -> "IndexKeyCache":
        return NPUBf16IndexKeyCache(self, self.index_buf_size)

    def _init_kpool_compress_tail_buffers(self, *args, **kwargs) -> None:
        """Add one spare request row to each tail ring.

        The decode writer is branch-free so that it holds no host
        synchronisation, which means masked-off rows still take part in the
        scatter. Their request index is clamped into range and would otherwise
        alias a live request -- a padded graph row usually carries
        ``req_pool_indices == 0`` -- and a duplicated destination makes the
        write order undefined, so the live row's update can be the one that
        loses. A spare row gives them somewhere that collides with nothing.
        """
        super()._init_kpool_compress_tail_buffers(*args, **kwargs)
        if not getattr(self, "kpool_use_compress", False):
            return
        pad = lambda t: (  # noqa: E731
            t
            if t.shape[0] == 0
            else torch.cat([t, torch.zeros_like(t[:1])], dim=0)
        )
        self._tail_scratch_row = max(
            (t.shape[0] for t in self._compress_tail_k), default=0
        )
        self._compress_tail_k = [pad(t) for t in self._compress_tail_k]
        self._compress_tail_score = [pad(t) for t in self._compress_tail_score]

    @property
    def scratch_loc(self) -> int:
        """An index-cache slot no block table can name -- see _buffer_shape."""
        return (self.index_key_cache.buffer[0].shape[0] - 1) * self.page_size

    def set_index_k_bf16(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        """Scatter compressed pooled keys to their cache slots.

        ``loc`` is a flat slot index -- ``page * page_size + offset_in_page`` --
        matching the addressing the fp8 kernels compute inline.
        """
        buf = self.get_index_k_with_scale_buffer(layer_id)
        torch_npu.npu_scatter_nd_update_(
            buf.view(-1, 1, self.index_head_dim),
            loc.reshape(-1, 1).long(),
            index_k.reshape(-1, 1, self.index_head_dim).to(torch.bfloat16),
        )

    def kpool_decode_update_index_cache(
        self,
        layer_id: int,
        key: torch.Tensor,
        slot_score: torch.Tensor,
        ape: torch.Tensor,
        block_tables: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        round_scale: bool = False,
    ) -> None:
        """Advance the tail ring, and compress into the cache when a pool closes.

        A torch transcription of ``_kpool_decode_update_and_maybe_write_cache_kernel``
        minus its fp8 store: same validity test, same ring addressing, same
        substitution of the incoming token for the slot it has not been written
        to yet, and the same pooled-slot address arithmetic.

        Assumes one row per request, which decode guarantees. Two rows sharing a
        ``req_pool_index`` would race for the same ring slot here -- as two of the
        kernel's programs would.
        """
        from sglang.srt.hardware_backend.npu.attention.kpool_indexer_npu import (
            compress_pool_bf16,
        )

        assert (
            self.kpool_use_compress
        ), "kpool_decode_update_index_cache called when kpool compress is disabled"
        batch = key.shape[0]
        if batch == 0:
            return

        idx = layer_id - self.start_layer
        tail_k, tail_score = self._compress_tail_k[idx], self._compress_tail_score[idx]
        pool_size, tail_width = self.index_kpool, tail_k.shape[1]
        req_pool_size = tail_k.shape[0]

        req = req_pool_indices[:batch].long()
        pos = positions[:batch].long()
        valid = (
            (req >= 0)
            & (req < req_pool_size)
            & (out_cache_loc[:batch] != 0)
            & (pos >= 0)
            & (pos < seq_lens[:batch])
        )
        safe_req = req.clamp(0, req_pool_size - 1)
        safe_pos = pos.clamp(min=0)

        # A pool closes on its last slot; only then is there anything to compress.
        # Every row is compressed and a mask decides what lands, rather than
        # selecting the closing rows: `.nonzero()` would move the count to the
        # host, which costs a synchronisation per layer per decode step and makes
        # the path impossible to capture into a graph. The batch is at most
        # max_running_requests, so compressing all of it is cheaper than the sync.
        closing = (valid & (safe_pos % pool_size == pool_size - 1)).unsqueeze(1)
        rows = torch.arange(batch, device=key.device)

        start = safe_pos - safe_pos % pool_size
        phys = (
            start.unsqueeze(1) + torch.arange(pool_size, device=key.device).unsqueeze(0)
        ) % tail_width
        # Flatten to one index tensor rather than indexing [req, slot] with two.
        # Multi-tensor advanced indexing has no AI Core implementation, so it
        # falls back to aclnnIndex on the AI *CPU*: profiled at 293-308us twice
        # per decode step -- 37.5% of this layer's whole device time -- to move
        # 35 KB. index_select on the flat view is the same gather on the AI Core.
        flat_k = tail_k.view(-1, tail_k.shape[-1])
        flat_s = tail_score.view(-1, tail_score.shape[-1])
        gather = (safe_req.unsqueeze(1) * tail_width + phys).reshape(-1)
        slot_k = flat_k.index_select(0, gather).view(batch, pool_size, -1)
        slot_s = flat_s.index_select(0, gather).view(batch, pool_size, -1)
        # The closing token is still in flight -- the ring is written below.
        slot_k[:, pool_size - 1] = key
        slot_s[:, pool_size - 1] = slot_score

        pool_id = safe_pos // pool_size
        page_col = ((pool_id // self.slots_per_page) * pool_size).clamp(
            0, block_tables.shape[1] - 1
        )
        page = block_tables[rows, page_col].long()
        loc = torch.where(
            closing.squeeze(1),
            page * self.page_size + pool_id % self.slots_per_page,
            torch.full_like(page, self.scratch_loc),
        )
        # A row that is not closing, or is invalid, is sent to the spare slot
        # instead of being filtered out -- filtering needs the count on the host.
        self.set_index_k_bf16(layer_id, loc, compress_pool_bf16(slot_k, slot_s, ape))

        # Same for the ring, and the write side takes the same treatment: a
        # two-tensor index_put_ is the AI CPU path again.
        dest = torch.where(valid, safe_req, self._tail_scratch_row)
        scatter = (dest * tail_width + safe_pos % tail_width).reshape(-1, 1)
        width = tail_k.shape[-1]
        torch_npu.npu_scatter_nd_update_(
            flat_k.view(-1, 1, width), scatter, key.reshape(-1, 1, width)
        )
        torch_npu.npu_scatter_nd_update_(
            flat_s.view(-1, 1, width), scatter, slot_score.reshape(-1, 1, width)
        )

    def set_index_k_scale_buffer(self, *args, **kwargs):
        raise NotImplementedError(
            "This pool stores index keys as bf16; there is no fp8 key + fp32 scale "
            "to write. Use set_index_k_bf16."
        )

    def get_index_k_scale_buffer(self, *args, **kwargs):
        raise NotImplementedError(
            "This pool stores index keys as bf16; there is no separate scale to read."
        )

    def get_index_k_scale_continuous(self, *args, **kwargs):
        raise NotImplementedError(
            "This pool stores index keys as bf16; there is no separate scale to read."
        )
