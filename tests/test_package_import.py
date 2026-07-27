from importlib import import_module


def test_package_boundaries_are_importable() -> None:
    modules = (
        "mini_llm_runtime",
        "mini_llm_runtime.data",
        "mini_llm_runtime.data.batching",
        "mini_llm_runtime.data.bpe",
        "mini_llm_runtime.data.tokenizer",
        "mini_llm_runtime.distributed",
        "mini_llm_runtime.kernels",
        "mini_llm_runtime.kernels.reference",
        "mini_llm_runtime.kernels.triton",
        "mini_llm_runtime.model",
        "mini_llm_runtime.model.attention",
        "mini_llm_runtime.model.rope",
        "mini_llm_runtime.model.transformer",
        "mini_llm_runtime.runtime",
        "mini_llm_runtime.training",
        "mini_llm_runtime.training.adamw",
        "mini_llm_runtime.training.checkpoint",
        "mini_llm_runtime.training.grad_clip",
        "mini_llm_runtime.training.losses",
        "mini_llm_runtime.training.schedules",
    )

    for module in modules:
        import_module(module)
