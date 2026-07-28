import torch

from torch.profiler import profile, ProfilerActivity

from mini_llm_runtime.kernels.triton.flash_attention import flash_attention


def make_inputs(batch_size, num_heads, seq_len, head_dim):
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


def profile_triton(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
):
    q, k, v = make_inputs(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    )

    # Triton FlashAttention 接口需要 [BH, S, D]
    B, H, S, D = q.shape

    q_triton = q.reshape(B * H, S, D)
    k_triton = k.reshape(B * H, S, D)
    v_triton = v.reshape(B * H, S, D)

    fn_triton = lambda: flash_attention(
        q_triton,
        k_triton,
        v_triton,
        is_causal=False,
    )

    # -------------------------
    # 1. Warmup
    # -------------------------
    for _ in range(10):
        fn_triton()

    torch.cuda.synchronize()

    # -------------------------
    # 2. Profile 一次 forward
    # -------------------------
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
    ) as prof:
        fn_triton()

    torch.cuda.synchronize()

    # -------------------------
    # 3. 输出结果
    # -------------------------
    print()
    print("=" * 80)
    print(
        f"Triton FlashAttention profile | "
        f"B={B}, H={H}, S={S}, D={D}"
    )
    print("=" * 80)

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )


def main():
    batch_size = 1
    num_heads = 8
    head_dim = 64

    # 先只看两个有代表性的点
    seq_lens = [512, 2048]

    for seq_len in seq_lens:
        profile_triton(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )


if __name__ == "__main__":
    main()