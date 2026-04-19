import json
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict
from vllm import SamplingParams

from src.executor import LocalExecutor
from src.metrics import BaseCodeMetric, ExecutionResult
from src.data.types import CodingTask
import src.config as config

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
            sampling_params = group['params']
            metrics_in_group = group['metrics']

            print(f"\n🚀 Group: {[m.name for m in metrics_in_group]}")
            print(f"⚙️ Params: n={sampling_params.n}, temp={sampling_params.temperature}, "
                  f"max_tokens={sampling_params.max_tokens}, logprobs={sampling_params.logprobs}")

            prompts = [t.prompt for t in tasks]
            outputs = self.llm.generate(prompts, sampling_params)

            flat_tasks_input = []     # [(code, tests), ...]
            map_indices = []          # [(task_idx, sample_idx), ...] для восстановления структуры

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
                entropy = self._calculate_entropy(original_sample)
                exec_res.entropy = entropy

                all_exec_results[task_idx].append(exec_res)

            for metric in metrics_in_group:
                score = metric.calculate(all_exec_results)
                final_results[metric.name] = score
                print(f"📊 {metric.name}: {score:.4f}")

        return final_results

    def _group_metrics_and_prepare_params(self):
        groups = {}
        base_settings = config.SAMPLING_SETTINGS.copy()
        base_settings.pop("n", None)
        base_settings.pop("temperature", None)

        for metric in self.metrics:
            cfg = metric.gen_config
            cfg_key = json.dumps(cfg, sort_keys=True)

            if cfg_key not in groups:
                n = cfg.get("num_return_sequences", 1)
                temp = cfg.get("temperature", 0.0)

                vllm_params = SamplingParams(
                    n=n,
                    temperature=temp,
                    stop_token_ids=[self.tokenizer.eos_token_id], # Важно для остановки
                    **base_settings
                )

                groups[cfg_key] = {'params': vllm_params, 'metrics': []}

            groups[cfg_key]['metrics'].append(metric)

        return groups

    def _calculate_entropy(self, sample_output):
        """Считает энтропию для vLLM outputs"""
        if not sample_output.logprobs:
            return 0.0

        entropies = []
        for step_logprobs in sample_output.logprobs:
            if not step_logprobs: continue

            val = list(step_logprobs.values())[0].logprob
            entropies.append(-val)

        return np.mean(entropies) if entropies else 0.0
