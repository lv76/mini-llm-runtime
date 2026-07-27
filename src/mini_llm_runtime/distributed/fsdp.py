from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from mini_llm_runtime.model.layers import Embedding, Linear


def _is_shardable_module(module: nn.Module) -> bool:
    return isinstance(module, (Linear, Embedding))


class FSDP(nn.Module):
    """
    简化版 FSDP

    当前实现规则：
    1. Linear.weight 和 Embedding.weight 被分片。
    2. 其他参数保持 replicated。
    3. sharded parameter:
       - forward 前 all-gather
       - forward 后 reshard
       - backward 前再次 all-gather
       - 梯度使用 reduce-scatter
    4. replicated parameter:
       - 梯度使用 all-reduce
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError(
                "The distributed process group must be initialized "
                "before constructing FSDP."
            )

        self.module = module
        self.compute_dtype = compute_dtype

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 每一个元素是一个字典，记录一个分片参数的全部信息。
        self.sharded_param_infos: list[dict] = []

        # 不分片的参数，例如 RMSNorm.weight。
        self.replicated_param_infos: list[dict] = []

        # 保存异步 reduce-scatter / all-reduce 通信任务。
        self._pending_grad_syncs: list[dict] = []

        # 可选：保证各个 rank 的初始化参数相同。
        self._sync_initial_parameters()

        # 将 Linear / Embedding 的 weight 切成本地 shard。
        self._shard_parameters()

        # 注册 forward、backward、gradient hooks。
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ============================================================
    # 1. 初始化阶段
    # ============================================================

    def _sync_initial_parameters(self) -> None:
        """
        初始化同步。

        注意：
        这里的 broadcast 只发生在参数分片之前。
        它不是 FSDP forward 时恢复完整权重的方法。

        forward 恢复权重使用的是 all-gather。
        """
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter.data, src=0)

    def _shard_parameters(self) -> None:
        """
        找出 Linear / Embedding 的 weight,并根据 rank 切片。

        完整参数：
            [original_numel]

        padding 后：
            [shard_numel * world_size]

        rank r 保存：
            [r * shard_numel : (r + 1) * shard_numel]
        """
        sharded_param_ids: set[int] = set()

        for module_name, submodule in self.module.named_modules():
            if not _is_shardable_module(submodule):
                continue

            parameter = submodule.weight

            # 防止同一个 Parameter 被重复分片。
            if id(parameter) in sharded_param_ids:
                continue

            parameter_name = (
                f"{module_name}.weight"
                if module_name
                else "weight"
            )

            original_shape = parameter.shape
            original_numel = parameter.numel()

            # ceil(original_numel / world_size)
            shard_numel = (
                original_numel + self.world_size - 1
            ) // self.world_size

            padded_numel = shard_numel * self.world_size
            padding_numel = padded_numel - original_numel

            full_flat_parameter = (
                parameter.detach()
                .reshape(-1)
                .contiguous()
            )

            if padding_numel > 0:
                full_flat_parameter = torch.cat(
                    [
                        full_flat_parameter,
                        full_flat_parameter.new_zeros(
                            padding_numel
                        ),
                    ],
                    dim=0,
                )

            start = self.rank * shard_numel
            end = start + shard_numel

            local_shard = (
                full_flat_parameter[start:end]
                .clone()
                .contiguous()
            )

            # 保留原 Parameter 对象，只替换底层数据。
            parameter.data = local_shard

            param_info = {
                "name": parameter_name,
                "module": submodule,
                "parameter": parameter,
                "original_shape": original_shape,
                "original_numel": original_numel,
                "shard_numel": shard_numel,
                "padded_numel": padded_numel,

                # 这是长期保存的 FP32 master shard。
                # optimizer 最终更新的就是它。
                "local_shard": local_shard,
            }

            self.sharded_param_infos.append(param_info)
            sharded_param_ids.add(id(parameter))

        # 其余参数保持 replicated。
        for parameter_name, parameter in (
            self.module.named_parameters()
        ):
            if id(parameter) in sharded_param_ids:
                continue

            self.replicated_param_infos.append(
                {
                    "name": parameter_name,
                    "parameter": parameter,
                }
            )

    # ============================================================
    # 2. 参数 all-gather
    # ============================================================

    def _all_gather_full_parameter(
        self,
        param_info: dict,
        *,
        use_compute_dtype: bool,
    ) -> torch.Tensor:
        """
        从所有 rank 收集参数 shard，恢复完整参数。

        use_compute_dtype=True:
            forward/backward 计算使用。
            如果 compute_dtype=float16，则通信 FP16 shard。

        use_compute_dtype=False:
            gather_full_params 使用。
            通信并返回 FP32 master 参数。
        """
        parameter = param_info["parameter"]

        if use_compute_dtype and self.compute_dtype is not None:
            communication_dtype = self.compute_dtype
        else:
            communication_dtype = parameter.data.dtype

        local_shard = (
            param_info["local_shard"]
            .to(dtype=communication_dtype)
            .contiguous()
        )

        gathered_shards = [
            torch.empty_like(local_shard)
            for _ in range(self.world_size)
        ]

        dist.all_gather(
            gathered_shards,
            local_shard,
        )

        full_flat_parameter = torch.cat(
            gathered_shards,
            dim=0,
        )

        # 去掉初始化时增加的 padding。
        full_flat_parameter = full_flat_parameter[
            : param_info["original_numel"]
        ]

        full_parameter = full_flat_parameter.view(
            param_info["original_shape"]
        )

        return full_parameter

    # ============================================================
    # 3. 注册 hooks
    # ============================================================

    def _register_hooks(self) -> None:
        for param_info in self.sharded_param_infos:
            submodule = param_info["module"]
            parameter = param_info["parameter"]

            # Forward 前恢复完整权重。
            submodule.register_forward_pre_hook(
                self._make_forward_pre_hook(param_info)
            )

            # Forward 后恢复本地 shard。
            submodule.register_forward_hook(
                self._make_forward_post_hook(param_info)
            )

            # Backward 前再次恢复完整权重。
            submodule.register_full_backward_pre_hook(
                self._make_backward_pre_hook(param_info)
            )

            # 完整局部梯度产生后执行 reduce-scatter。
            parameter.register_post_accumulate_grad_hook(
                self._make_sharded_grad_hook(param_info)
            )

        # replicated 参数只需要同步梯度。
        for param_info in self.replicated_param_infos:
            parameter = param_info["parameter"]

            if not parameter.requires_grad:
                continue

            parameter.register_post_accumulate_grad_hook(
                self._make_replicated_grad_hook(param_info)
            )

    # ============================================================
    # 4. Forward hooks
    # ============================================================

    def _make_forward_pre_hook(self, param_info: dict):
        def hook(module: nn.Module, inputs):
            parameter = param_info["parameter"]

            # 此时 parameter.data 应该是本地 FP32 shard。
            full_parameter = self._all_gather_full_parameter(
                param_info,
                use_compute_dtype=True,
            )

            # 临时安装完整 FP32/FP16 权重。
            parameter.data = full_parameter

        return hook

    def _make_forward_post_hook(self, param_info: dict):
        def hook(module: nn.Module, inputs, output):
            parameter = param_info["parameter"]

            # Forward 结束，重新恢复 FP32 master shard。
            parameter.data = param_info["local_shard"]

            # 确保 backward 时重新创建与完整参数匹配的梯度。
            parameter.grad = None

            return output

        return hook

    # ============================================================
    # 5. Backward pre-hook
    # ============================================================

    def _make_backward_pre_hook(self, param_info: dict):
        def hook(module: nn.Module, grad_output):
            parameter = param_info["parameter"]

            # Backward 计算 dX 时需要完整权重。
            full_parameter = self._all_gather_full_parameter(
                param_info,
                use_compute_dtype=True,
            )

            parameter.data = full_parameter
            parameter.grad = None

        return hook

    # ============================================================
    # 6. Sharded gradient hook
    # ============================================================

    def _make_sharded_grad_hook(self, param_info: dict):
        def hook(parameter: nn.Parameter):
            if parameter.grad is None:
                return

            local_master_shard = param_info["local_shard"]

            # 当前 parameter.grad 是完整的局部梯度。
            full_local_grad = parameter.grad.detach()

            # optimizer 更新 FP32 master shard，
            # 因此梯度最终也转为 FP32。
            full_flat_grad = (
                full_local_grad
                .reshape(-1)
                .to(dtype=local_master_shard.dtype)
                .contiguous()
            )

            padding_numel = (
                param_info["padded_numel"]
                - param_info["original_numel"]
            )

            if padding_numel > 0:
                full_flat_grad = torch.cat(
                    [
                        full_flat_grad,
                        full_flat_grad.new_zeros(
                            padding_numel
                        ),
                    ],
                    dim=0,
                )

            local_grad_shard = torch.empty(
                param_info["shard_numel"],
                device=full_flat_grad.device,
                dtype=local_master_shard.dtype,
            )

            # 每个 rank 输入完整局部梯度。
            # 每个 rank 输出对应位置的梯度 shard。
            work = dist.reduce_scatter_tensor(
                local_grad_shard,
                full_flat_grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            # 先清除完整梯度，避免 shape/dtype 校验冲突。
            parameter.grad = None

            # 恢复本地 FP32 参数 shard。
            parameter.data = local_master_shard

            # 本地 grad shard 的内容将在通信完成后有效。
            parameter.grad = local_grad_shard

            self._pending_grad_syncs.append(
                {
                    "kind": "sharded",
                    "parameter": parameter,
                    "work": work,

                    # async 通信结束前，输入 tensor 必须保持存活。
                    "communication_input": full_flat_grad,
                }
            )

        return hook

    # ============================================================
    # 7. Replicated gradient hook
    # ============================================================

    def _make_replicated_grad_hook(self, param_info: dict):
        def hook(parameter: nn.Parameter):
            if parameter.grad is None:
                return

            grad = (
                parameter.grad.detach()
                .to(dtype=parameter.data.dtype)
                .contiguous()
            )

            parameter.grad = grad

            work = dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            self._pending_grad_syncs.append(
                {
                    "kind": "replicated",
                    "parameter": parameter,
                    "work": work,
                }
            )

        return hook

    # ============================================================
    # 8. 等待梯度同步完成
    # ============================================================

    def finish_gradient_synchronization(self) -> None:
        """
        backward 后、optimizer.step 前调用。

        作用：
        1. 等待 reduce-scatter / all-reduce 完成。
        2. 将梯度和除以 world_size，得到平均梯度。
        """
        for pending in self._pending_grad_syncs:
            pending["work"].wait()

            parameter = pending["parameter"]

            if parameter.grad is not None:
                parameter.grad.div_(self.world_size)

        self._pending_grad_syncs.clear()

    # ============================================================
    # 9. 恢复完整参数字典
    # ============================================================

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """
        返回底层模型的完整参数字典。

        Sharded 参数：
            all-gather FP32 master shards。

        Replicated 参数：
            直接复制当前完整参数。
        """
        result: dict[str, torch.Tensor] = {}

        sharded_param_ids = {
            id(info["parameter"])
            for info in self.sharded_param_infos
        }

        for param_info in self.sharded_param_infos:
            full_parameter = self._all_gather_full_parameter(
                param_info,
                use_compute_dtype=False,
            )

            result[param_info["name"]] = (
                full_parameter.detach().clone()
            )

        for name, parameter in self.module.named_parameters():
            if id(parameter) in sharded_param_ids:
                continue

            result[name] = parameter.detach().clone()

        return result
