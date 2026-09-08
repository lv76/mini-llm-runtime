import argparse

import torch

from mini_llm_runtime.kernels.triton.flash_attention import flash_fwd_triton


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile the BM16 or BM64 Triton FlashAttention "
            "forward configuration with Nsight Compute."
        )
    )
    parser.add_argument(
        "--block-m",
        type=int,
        choices=(16, 64),
        required=True,
        help="Q tile size to profile; supported values are 16 and 64.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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

    B, H, S, D = q.shape

    q_triton = q.reshape(B * H, S, D)
    k_triton = k.reshape(B * H, S, D)
    v_triton = v.reshape(B * H, S, D)

    def run():
        out, _ = flash_fwd_triton(
            q_triton,
            k_triton,
            v_triton,
            is_causal=False,
            block_m=args.block_m,
            block_n=64,
            num_warps=4,
            num_stages=2,
        )
        return out

    for _ in range(10):
        run()

    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    out = run()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"BLOCK_M={args.block_m}")
    print(f"output shape={tuple(out.shape)}")
    print(f"output dtype={out.dtype}")


if __name__ == "__main__":
    main()
