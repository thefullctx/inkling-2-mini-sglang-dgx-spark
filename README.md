# Inkling-2-Mini on a Single NVIDIA DGX Spark (SGLang + DSpark, Tuned)

Tuned, single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **DSpark speculative decoding** on a **single NVIDIA DGX Spark** using **SGLang**.

> **Why SGLang won our DGX Spark A/B test & tuning sweep:** After benchmarking SGLang vs. vLLM and sweeping 12 SM121/Blackwell SGLang configurations, our tuned SGLang recipe delivers **+19.4% faster 15.6K-context decode** (`39.99` → `47.75 tok/s`), **-23% faster warm prefix TTFT** (`0.27s` → `0.21s`), **2.06× faster long-context prefill than vLLM**, and uses **~43 GiB less unified RAM than vLLM**—while passing chat, reasoning, tool calling, and vision out of the box.
> Looking for the vLLM recipe? See [inkling-2-mini-vllm-dgx-spark](https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark).

---

## Hardware & Model Overview

* **Hardware:** 1× NVIDIA DGX Spark (128 GB unified LPDDR5X, Blackwell native FP4)
* **Model:** `thinkingmachines/Inkling-2-Mini` — 26.4B total / 3.4B active (~5.4B effective across 3 looped middle-layer passes), multimodal (text + image in, text out), up to 1M context, bundled DSpark speculator
* **Precision:** Native mixed **NVFP4** (layers 0–19 experts) + **MXFP8** (layers 20–23 experts) + **BF16** (attention, vision, draft head & KV cache) — ~22 GB total weights, zero requantization
* **Image:** `lmsysorg/sglang:inkling-2-mini` (`linux/arm64` native)

---

## Key Tuning Wins Over the Default SGLang Recipe

On a single DGX Spark (memory-bandwidth bound at ~273 GB/s LPDDR5X), three targeted tweaks unlocked the biggest real gains over the default cookbook command:

1. **`--page-size 16`** (instead of `128`): Eliminates KV page-tail waste across Inkling-2-Mini's 20 sliding-window attention layers and RadixAttention prefix boundaries → **+19.4% 15.6K-context decode**, **-23% warm prefix TTFT**, and **fastest cold prefill (`4.47s`)**.
2. **`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1` + `SGLANG_OPT_MOE_QUANT_ONCE=1`**: Fuses MoE finalize and avoids redundant activation quantization → **+3.2% `c=4` aggregate throughput** with zero extra memory.
3. **`--cuda-graph-max-bs-decode 8`**: Caps decode CUDA-graph capture for single-Spark workloads, saving unified memory without hurting throughput.

---

## Deterministic Tuning Sweep (`temperature=0.0`, 1× DGX Spark)

Measured with [`benchmarks/bench.py`](benchmarks/bench.py) at `temperature=0.0` (zero DSpark sampling variance) over `/v1/chat/completions`:

| Experiment | `c=1` Short | 15.6K Cold | 15.6K Warm | 15.6K Decode | `c=4` Agg | RAM | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `t0-00-baseline` *(default page=128)* | 45.22 t/s | 4.55s | 0.27s | 39.99 t/s | 87.63 t/s | 62.7G | Official cookbook defaults |
| `t0-01-moe-fused-quant` | **45.35 t/s** | 4.54s | 0.27s | 40.08 t/s | 89.16 t/s | 62.9G | Fused MoE finalize + quant once |
| `t0-09-page64` *(+ fused MoE)* | 45.23 t/s | 4.53s | **0.20s** | 47.70 t/s | 88.61 t/s | 62.3G | Huge long-ctx jump |
| `t0-11-page32` *(+ fused MoE)* | 45.22 t/s | 4.55s | 0.21s | 47.73 t/s | 84.72 t/s | 62.0G | Long-ctx ceiling reached |
| **`t0-12-page16` *(final recipe)*** | 45.16 t/s | **4.47s** | 0.21s | **47.75 t/s** | **90.44 t/s** | **62.8G** | **Best overall across all workloads** |
| `t0-04-moe-trtllm` | 48.41 t/s | 8.78s | 0.32s | 33.16 t/s | 62.06 t/s | 92.0G | Faster short `c=1`, but 2× slower prefill & -29% `c=4` |
| `t0-05-bf16-gemv` | 44.62 t/s | 4.56s | 0.27s | 39.45 t/s | 90.60 t/s | 82.8G | +20G RAM, slower `c=1` |

*(Also tested and ruled out on SM121: `--kv-cache-dtype fp8_e4m3` unsupported by FA4 SM120, `--inkling2-loop-share-cache` requires `is_varloop=true`, `--bf16-gemm-backend cutedsl` requires SM100/SM103, `--moe-runner-backend flashinfer_cutedsl` unsupported for `inkling2` quant, `SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=1` regressed long-context decode to 36.4 tok/s, and DSpark enforces `draft_tokens == 8`.)*

### Head-to-Head vs. vLLM (`temperature=0.6`, 1× DGX Spark)

| Metric | **SGLang + DSpark** | vLLM + DSpark | Winner |
| :--- | :---: | :---: | :---: |
| **Smoke Tests** (Chat/Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | Tie |
| **`c=1` Short Decode (3-run avg)** | 49.56 tok/s *(peak 64.4)* | **52.30 tok/s** | vLLM (+5.5%) |
| **`c=1` 15.6K-Context Cold Prefill** | **4.47s** *(~3,480 tok/s)* | 9.23s *(~1,690 tok/s)* | **SGLang (2.06×)** |
| **`c=1` 15.6K-Context Warm Prefix TTFT** | **0.21s** *(tuned)* | 0.23s | **SGLang** |
| **`c=1` 15.6K-Context Decode** | **47.75 tok/s** *(tuned)* | 38.43 tok/s | **SGLang (+24.3%)** |
| **`c=4` Concurrent Aggregate** | **90.4–94.2 tok/s** *(tuned)* | 91.17 tok/s | **Tie / SGLang** |
| **Unified RAM Used** | **62.8 GiB** *(~59 GiB free)* | 105.5 GiB *(16.1 GiB free)* | **SGLang** |

Raw JSON outputs for every single run are in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

### Option 1: One-command launch (`start.sh` / Docker Compose)

```bash
git clone https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark.git
cd inkling-2-mini-sglang-dgx-spark
cp .env.example .env
./start.sh
```

Follow logs:

```bash
docker logs -f inkling2-mini-sglang
```

Stop:

```bash
./stop.sh
```

### Option 2: `sparkrun`

```bash
sparkrun run sparkrun.yaml
```

### Option 3: Persistent `systemd` user service (survives reboots & crashes)

```bash
mkdir -p ~/.config/systemd/user
cp systemd/inkling2-mini-sglang.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now inkling2-mini-sglang.service
loginctl enable-linger "$USER"
```

---

## Verify & Benchmark

```bash
# Deterministic tuning comparison (temperature=0.0)
python3 benchmarks/bench.py --label my-run --temperature 0.0

# Realistic sampling benchmark (temperature=0.6)
python3 benchmarks/bench.py --label my-run-t06 --temperature 0.6
```
