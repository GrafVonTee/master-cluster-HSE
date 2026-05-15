import json
import numpy as np
from typing import List, Dict

from src.executor import LocalExecutor
from src.metrics import BaseCodeMetric
from src.data.types import CodingTask
from src.inference.vllm_inference import make_sampling_params


class Evaluator:
    def __init__(self, llm_engine, tokenizer, metrics: List[BaseCodeMetric]):
        self.llm = llm_engine
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.executor = LocalExecutor()

    def run(self, tasks: List[CodingTask]) -> Dict[str, float]:
        final_results = {}
        grouped_configs = self._group_metrics_and_prepare_params()

        for config_key, group in grouped_configs.items():
            sampling_params = group["params"]
            metrics_in_group = group["metrics"]

            print(f"\n🚀 Group: {[m.name for m in metrics_in_group]}")
            print(
                f"⚙️ Params: n={sampling_params.n}, temp={sampling_params.temperature}, "
                f"max_tokens={sampling_params.max_tokens}, logprobs={sampling_params.logprobs}"
            )

            prompts = [t.prompt for t in tasks]
            outputs = self.llm.generate(prompts, sampling_params)

            flat_tasks_input = []
            map_indices = []

            for i, request_output in enumerate(outputs):
                task_data = tasks[i]
                for j, sample in enumerate(request_output.outputs):
                    flat_tasks_input.append((sample.text, task_data.tests))
                    map_indices.append((i, j))

            flat_results = self.executor.batch_execute(flat_tasks_input)
            all_exec_results = [[] for _ in range(len(tasks))]

            for k, exec_res in enumerate(flat_results):
                task_idx, sample_idx = map_indices[k]
                original_sample = outputs[task_idx].outputs[sample_idx]
                exec_res.entropy = self._calculate_entropy(original_sample)
                all_exec_results[task_idx].append(exec_res)

            for metric in metrics_in_group:
                score = metric.calculate(all_exec_results)
                final_results[metric.name] = score
                print(f"📊 {metric.name}: {score:.4f}")

        return final_results

    def _group_metrics_and_prepare_params(self):
        groups = {}

        for metric in self.metrics:
            cfg = metric.gen_config
            cfg_key = json.dumps(cfg, sort_keys=True)

            if cfg_key not in groups:
                n = cfg.get("num_return_sequences", 1)
                temp = cfg.get("temperature", 0.0)

                # Important: make_sampling_params always sets max_tokens explicitly
                # from EVAL_MAX_NEW_TOKENS. vLLM default is only 16.
                vllm_params = make_sampling_params(
                    self.tokenizer,
                    n=n,
                    temperature=temp,
                    logprobs=1,
                )

                groups[cfg_key] = {"params": vllm_params, "metrics": []}

            groups[cfg_key]["metrics"].append(metric)

        return groups

    def _calculate_entropy(self, sample_output):
        if not sample_output.logprobs:
            return 0.0

        entropies = []
        for step_logprobs in sample_output.logprobs:
            if not step_logprobs:
                continue
            val = list(step_logprobs.values())[0].logprob
            entropies.append(-val)

        return float(np.mean(entropies)) if entropies else 0.0
