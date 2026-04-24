import os
from functools import partial
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import src.config as config

def setup_model(model_path=config.MODEL_PATH, adapter_path=None):
    using_sft = adapter_path is not None
    llm = LLM(
        model=model_path,
        enable_lora=using_sft,
        max_lora_rank=64 if using_sft else None,
        **config.VLLM_PARAMS
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if using_sft else model_path,
        trust_remote_code=True
    )

    if using_sft:
        request = LoRARequest("sft", 1, lora_path=adapter_path)
        llm.generate = partial(llm.generate, lora_request=request)

    sampling_params = SamplingParams(
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        **config.SAMPLING_SETTINGS
    )

    return llm, tokenizer, sampling_params
