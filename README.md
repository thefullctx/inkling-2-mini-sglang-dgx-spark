# Inkling-2-Mini on a Single NVIDIA DGX Spark (SGLang + DFlash2)

Fast, lightweight, **lossless** single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)** speculative decoding on a **single NVIDIA DGX Spark** using the official **SGLang** image (zero C++/CUDA compilation).

> **Highlights vs. Stock SGLang + DSpark (1× DGX Spark, lossless target verification):**
> * **`c=1` short decode (`temp=0`):** `45.2` → **61.8 tok/s (+36.6%)** *(+22.6% over our previous tuned DSpark)*
> * **`c=1` short decode (`temp=0.6`):** `49.6` → **56.9 tok/s (+14.7%)**
> * **15.6K long-context decode (`temp=0`):** `40.0` → **61.1 tok/s (+52.8%)** — virtually zero slowdown vs. short context!
> * **`c=4` concurrent aggregate (`temp=0`):** `87.6` → **122.3 tok/s (+39.6%)** | **113.2 tok/s** at `temp=0.6`
> * **Unified RAM headroom:** **~66.8 GiB used / ~54 GiB free** on 121.6 GiB Spark *(4.3 GiB less than our tuned DSpark setup, easily coexists with a second small model)*
> * **Lossless quality:** Unmodified 3-pass Inkling-2-Mini target verifies every token (no target weight/sampling changes); reasoning + tool calling + vision all pass ✅.
>
> Looking for our vLLM recipe? See [inkling-2-mini-vllm-dgx-spark](https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark).

---

## What Changed Over Stock SGLang

1. **DFlash 2 Speculative Drafter (`incoai/Inkling-2-Mini-DFlash2` + `overlay/sglang_inkling_dflash2.py`):**
   * Replaces built-in DSpark with Inco AI's 0.5B (`990 MB`, Apache-2.0) **DFlash 2** block-diffusion drafter (`--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Inkling-2-Mini-DFlash2 --speculative-draft-attention-backend fa4 --speculative-num-draft-tokens 8`).
   * Applies Inco AI's official `sglang_inkling_dflash2.py` compatibility patch at container startup (wires Inkling's `[5, 23, 41, 48]` feature taps, `selector_softcap_mode: "after_correction"`, and real vocab size `200,058` — zero target math, sampling, or weight changes).
2. **DGX Spark Target Optimizations:**
   * `--page-size 16` (eliminates SWA/KV page fragmentation at long context), `overlay/inkling2_moe.py` (fused MoE quantization path), `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1`, `SGLANG_OPT_MOE_QUANT_ONCE=1`, `--mem-fraction-static 0.45`, `--cuda-graph-max-bs-decode 8`.

---

## Benchmarks (`1× NVIDIA DGX Spark`, All Smoke Tests ✅)

Measured apples-to-apples with [`benchmarks/bench.py`](benchmarks/bench.py) over `/v1/chat/completions`:

| Metric | Stock SGLang + DSpark (`page=128`) | Tuned (`page=16`) | Previous Tuned DSpark (`INT8 + vcap5`) | **SGLang + DFlash2 (`fa4 + page=16`)** | Gain vs. Stock |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Smoke Tests** (Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | **✅ / ✅ / ✅** | — |
| **`c=1` Short Decode (`temp=0.0`)** | 45.22 tok/s | 45.16 tok/s | 50.37 tok/s | **61.75 tok/s** | **+36.6%** |
| **`c=1` Short Decode (`temp=0.6`)** | 49.56 tok/s | — | 53.91 tok/s | **56.86 tok/s** | **+14.7%** |
| **15.6K Cold Prefill** | 4.55s | 4.47s | 4.49s | **4.45s** | **−2.2%** |
| **15.6K Warm Prefix TTFT** | 0.27s | 0.21s | 0.16s | **0.21s** | **−22.2%** |
| **15.6K Long-Context Decode (`temp=0.0`)** | 39.99 tok/s | 47.75 tok/s | 51.42 tok/s | **61.10 tok/s** | **+52.8%** |
| **15.6K Long-Context Decode (`temp=0.6`)** | — | — | 45.12 tok/s | **48.21 tok/s** | — |
| **`c=4` Concurrent (`temp=0.0`)** | 87.63 tok/s | 90.44 tok/s | 101.81 tok/s | **122.32 tok/s** | **+39.6%** |
| **`c=4` Concurrent (`temp=0.6`)** | 87.88 tok/s | — | 97.14 tok/s | **113.19 tok/s** | **+28.8%** |
| **Unified RAM Used** | 62.7 GiB | 62.8 GiB | 71.1 GiB | **66.8 GiB (~54 GiB free)** | +4.1 GiB |

Raw JSON logs for all 21 benchmark runs are in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

```bash
git clone https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark.git
cd inkling-2-mini-sglang-dgx-spark
cp .env.example .env
./start.sh
```

*(First launch downloads `thinkingmachines/Inkling-2-Mini` (~20 GB) and `incoai/Inkling-2-Mini-DFlash2` (~990 MB) into `~/.cache/huggingface` with `HF_HUB_DISABLE_XET=1`.)*

Check status / logs / stop:

```bash
./status.sh
./logs.sh
./stop.sh
```

### Persistent `systemd` User Service

```bash
mkdir -p ~/.config/systemd/user
cp systemd/inkling2-mini-sglang.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now inkling2-mini-sglang.service
loginctl enable-linger "$USER"
```

---

## Credits & License

* Apache-2.0 (`LICENSE`).
* **[Inco AI](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)** — trained and released the Apache-2.0 `Inkling-2-Mini-DFlash2` drafter and wrote the `sglang_inkling_dflash2.py` compatibility patch (`overlay/sglang_inkling_dflash2.py`).
* **[z-lab](https://github.com/z-lab/dflash)** (Jian Chen, Yesheng Liang, Zhijian Liu) — original DFlash block-diffusion speculative decoding architecture.
* **[tonyd2wild](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark)** — brought DFlash2 for Inkling 2 Mini to DGX Spark and benchmarked it across vLLM & TensorFold, sparking this SGLang + DFlash2 update.
* **[Thinking Machines Lab](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** — Inkling-2-Mini (Apache-2.0).
* **[SGLang](https://github.com/sgl-project/sglang)** — official `lmsysorg/sglang:inkling-2-mini` image, FA4, and DFlash2 worker.
* **[knapcio](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark)** — original 1× DGX Spark SGLang baseline flags (`--mem-fraction-static 0.45`, `--cuda-graph-max-bs-decode 8`).
* **[sxuff](https://github.com/sxuff/inkling-2-mini-spark)** — inspired our previous tuned DSpark baseline (INT8 draft + verify-row cap) included in the benchmark history above.
