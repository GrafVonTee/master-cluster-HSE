# HSE CL/RL Course Work

Первый milestone: запуск Qwen/Qwen3-14B на вычислительном кластере НИУ ВШЭ через Slurm.

## Что проверяем

- подключение к кластеру;
- создание conda-окружения;
- установку vLLM/transformers;
- запуск GPU-задачи через sbatch;
- генерацию короткого ответа моделью Qwen/Qwen3-14B.

## Локальный запуск структуры

```bash
mkdir scripts jobs logs outputs data
touch scripts/pelmeni_qwen14b.py
touch jobs/pelmeni_qwen14b.sbatch
touch logs/.gitkeep outputs/.gitkeep data/.gitkeep
```

## Установка окружения на кластере

```bash
module purge
module load Python

conda create -n clrl python=3.11 -y
source activate clrl

pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Запуск smoke-test

```bash
sbatch jobs/00_smoke_qwen14b.sbatch
```
