import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from mini_llm_runtime.model.qwen2 import (
    TinyQwen2Config,
    TinyQwen2ForCausalLM,
)
from mini_llm_runtime.model.qwen2_loader import (
    load_qwen2_safetensors,
)


MODEL_DIR = Path("checkpoints/Qwen2.5-0.5B")
CONFIG_PATH = MODEL_DIR / "config.json"
CHECKPOINT_PATH = MODEL_DIR / "model.safetensors"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison")

    device = torch.device("cuda")
    input_ids = torch.tensor(
        [[10, 20, 30, 40]],
        dtype=torch.long,
        device=device,
    )

    print("Loading Hugging Face model...")

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    hf_model.eval()

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits[0, -1].float().cpu()

    print("HF top token:", int(hf_logits.argmax().item()))

    del hf_model
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading local model...")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config_dict = json.load(file)

    config = TinyQwen2Config.from_dict(config_dict)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    try:
        with torch.device(device):
            local_model = TinyQwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    load_qwen2_safetensors(
        local_model,
        CHECKPOINT_PATH,
        tie_word_embeddings=config.tie_word_embeddings,
        device="cpu",
    )

    local_model.eval()

    with torch.no_grad():
        local_logits = local_model(input_ids)[0, -1].float().cpu()

    print("Local top token:", int(local_logits.argmax().item()))

    difference = (local_logits - hf_logits).abs()

    print("Max absolute difference:", difference.max().item())
    print("Mean absolute difference:", difference.mean().item())

    torch.testing.assert_close(
        local_logits,
        hf_logits,
        rtol=2e-2,
        atol=2e-2,
    )

    print("Dense logits alignment passed")


if __name__ == "__main__":
    main()
