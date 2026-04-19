from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-14B"


def main():
    print("Starting Qwen3-14B smoke test with vLLM...", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Кратко и по шагам объясни, как сварить пельмени. "
                "Ответь по-русски. /no_think"
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("Loading model with vLLM...", flush=True)

    llm = LLM(
		model=MODEL_NAME,
		dtype="half",
		max_model_len=2048,
		gpu_memory_utilization=0.95,
		trust_remote_code=True,
	)

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.8,
        max_tokens=512,
    )

    print("Generating answer...", flush=True)

    outputs = llm.generate([prompt], sampling_params)
    answer = outputs[0].outputs[0].text.strip()

    print("\n===== MODEL ANSWER =====\n", flush=True)
    print(answer, flush=True)
    print("\n===== END =====\n", flush=True)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "qwen14b_pelmeni_answer.txt"
    out_path.write_text(answer, encoding="utf-8")

    print(f"Saved answer to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
