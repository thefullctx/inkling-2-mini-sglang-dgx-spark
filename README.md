# Inkling-2-Mini on a Single NVIDIA DGX Spark (SGLang + DSpark + Lossless INT8 Draft)

Tuned, single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **DSpark speculative decoding** and a **custom lossless Triton INT8 draft kernel** on a **single NVIDIA DGX Spark** using **SGLang**.

> **What's new:** Adding a custom Triton weight-only per-row INT8 GEMV/GEMM overlay (`overlay/gemv_int8.py`) for the DSpark draft LM head (`201K × 2304`) and Markov fold (`201K × 256`, called 7× per step) cuts draft weight traffic in half on GB10's 273 GB/s LPDDR5X bus—delivering **+6.2% `c=1` decode**, **50.37 tok/s at 15.6K context (+26.0% vs stock SGLang!)**, **0.20s warm prefix TTFT**, and **94.87 tok/s at `c=4` (peak 111.3 tok/s)** with **5/5 byte-identical greedy outputs** (target model stays 100% native BF16/NVFP4/MXFP8).
> Looking for our vLLM recipe? See [inkling-2-mini-vllm-dgx-spark](https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark).

---

## Hardware & Model Overview

* **Hardware:** 1× NVIDIA DGX Spark (128 GB unified LPDDR5X, Blackwell native FP4)
* **Model:** `thinkingmachines/Inkling-2-Mini` — 26.4B total / 3.4B active (~5.4B effective across 3 looped middle-layer passes), multimodal (text + image in, text out), up to 1M context, bundled DSpark speculator
* **Precision:** Native mixed **NVFP4** (layers 0–19 experts) + **MXFP8** (layers 20–23 experts) + **BF16** (attention, vision, KV cache & target LM head); **weight-only INT8** on the DSpark draft LM head & Markov fold only (lossless verification!)
* **Image:** `lmsysorg/sglang:inkling-2-mini` (`linux/arm64` native)

---

## Key Optimizations Over Stock SGLang

1. **Custom Triton INT8 DSpark Draft Kernel (`overlay/gemv_int8.py`, `SGLANG_INKLING2_DRAFT_INT8=1`):**
   * Quantizes **only** the draft-side LM head copy (`201,024 × 2,304`) and Markov fold (`201,024 × 256`) to per-row symmetric INT8 at load time, saving ~825 MB of LPDDR5X reads every speculative step.
   * **Microbenchmarked on GB10:** Markov fold `452.6 µs → 218.7 µs` (**2.07×**); Draft LM head `3,902.5 µs → 2,007.6 µs` (**1.94×** at `bs=1`, **2.45×** at `bs=4`).
   * **100% lossless:** Target verification remains untouched in native BF16/NVFP4/MXFP8 (**5/5 byte-identical** greedy outputs; exact rejection sampling). Set `SGLANG_INKLING2_DRAFT_INT8=0` in `.env` to toggle off anytime.
2. **`--page-size 16`** (instead of `128`): Eliminates KV page-tail waste across Inkling-2-Mini's 20 sliding-window attention layers and RadixAttention prefix boundaries.
3. **`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1` + `SGLANG_OPT_MOE_QUANT_ONCE=1`**: Fuses MoE finalize and avoids redundant activation quantization.
4. **`--cuda-graph-max-bs-decode 8`**: Caps decode CUDA-graph capture for single-Spark workloads.

---

## Deterministic Benchmarks (`temperature=0.0`, 1× DGX Spark)

Measured with [`benchmarks/bench.py`](benchmarks/bench.py) at `temperature=0.0` (zero DSpark sampling variance) over `/v1/chat/completions`:

| Metric | Stock SGLang (`page=128`) | Tuned SGLang (`page=16`) | **+ INT8 Draft Overlay (this repo)** | Total Gain vs Stock |
| :--- | :---: | :---: | :---: | :---: |
| **Smoke Tests** (Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | **✅ / ✅ / ✅** | 100% Pass |
| **Greedy Output Match** | — | Ref | **5/5 Byte-Identical** ✅ | Zero drift |
| **`c=1` Short Decode (3-run avg)** | 45.22 tok/s | 45.16 tok/s | **47.97 tok/s** *(53.2 / 38.5 / 52.2)* | **+6.1%** |
| **15.6K Cold Decode** | — | 74.69 tok/s | **78.86 tok/s** | **+5.6% vs tuned** |
| **15.6K Warm Prefix TTFT** | 0.27s | 0.21s | **0.20s (198 ms)** | **-26%** |
| **15.6K Warm Decode** | 39.99 tok/s | 47.75 tok/s | **50.37 tok/s** | **+26.0%** |
| **`c=4` Concurrent Aggregate** | 87.63 tok/s | 90.44 tok/s | **94.87 tok/s** *(peak **111.3**)* | **+8.3%** |
| **Unified RAM Used** | 62.7 GiB | 62.8 GiB | **71.0 GiB** *(~50.6 GiB free)* | Plenty of headroom |

Raw JSON outputs for every experiment (14 total runs) are in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

### Option 1: One-command launch (`start.sh` / Docker Compose)

```bash
git clone https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark.git
cd inkling-2-mini-sglang-dgx-spark
cp .env.example .env
./start.sh
```

Check status / logs / stop:

```bash
./status.sh
./logs.sh
./stop.sh
```

### Option 2: `sparkrun`

```bash
sparkrun run sparkrun.yaml
```

*(Note: `sparkrun.yaml` runs the stock image with `--page-size 16` + fused MoE flags; use Docker Compose / `./start.sh` to enable the read-only `overlay/` INT8 draft kernel.)*

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
# Deterministic comparison (temperature=0.0)
python3 benchmarks/bench.py --label my-run --temperature 0.0

# Realistic sampling benchmark (temperature=0.6)
python3 benchmarks/bench.py --label my-run-t06 --temperature 0.6
```

## Unified Memory Budget (1× DGX Spark — 121.6 GiB Usable)

Measured live on `NVIDIA GB10` (`--mem-fraction-static 0.45`, `SGLANG_INKLING2_DRAFT_INT8=1`):

| Component | Footprint |
| :--- | :---: |
| Main weights (`Inkling2LoopForConditionalGeneration`, NVFP4/MXFP8) | 20.6 GiB |
| DSpark draft head (BF16) + INT8 draft LM head & Markov fold copies | ~1.9 GiB |
| Target + draft KV & Mamba state pools (~475K full tokens, 47K SWA) | 23.9 GiB |
| CUDA graphs, runtime buffers & host OS | ~24.6 GiB |
| **Total Used** | **~71.0 GiB** |
| **Free Headroom (OS / local agents / tools)** | **~50.6 GiB** |

## Credits & License

* Apache-2.0 (`LICENSE`).
* Hat tip to [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) for inspiring the lossless INT8 DSpark draft-head quantization approach (implemented here from scratch in Triton for SGLang's `Inkling2MTPHead`).
