import torch

from mini_llm_runtime.kernels.triton.flash_attention import flash_attention


def make_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    device = "cuda"
    dtype = torch.float16

    shape = (
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    )

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    return q, k, v


def main():
    # 固定一个有代表性的 workload
    batch_size = 1
    num_heads = 8
    seq_len = 2048
    head_dim = 64

    q, k, v = make_inputs(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    )

    # 你的 Triton FlashAttention 接口是 [BH, S, D]
    B, H, S, D = q.shape

    q_triton = q.reshape(B * H, S, D)
    k_triton = k.reshape(B * H, S, D)
    v_triton = v.reshape(B * H, S, D)

    # -------------------------
    # Warmup
    # -------------------------
    # 让 Triton 完成 JIT compile，并让 CUDA context 稳定下来
    for _ in range(10):
        flash_attention(
            q_triton,
            k_triton,
            v_triton,
            is_causal=False,
        )

    torch.cuda.synchronize()

    # -------------------------
    # Target kernel
    # -------------------------
    out = flash_attention(
        q_triton,
        k_triton,
        v_triton,
        is_causal=False,
    )

    torch.cuda.synchronize()

    # 防止脚本看起来什么都没做，同时简单确认输出
    print(
        f"Finished Triton FlashAttention: "
        f"shape={tuple(out.shape)}, "
        f"dtype={out.dtype}"
    )


if __name__ == "__main__":
    main()