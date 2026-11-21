# Inkling-2-Mini on a Single NVIDIA DGX Spark (SGLang + DSpark)

Production-ready, single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **DSpark speculative decoding** on a **single NVIDIA DGX Spark** using **SGLang**.

> **Why SGLang won our DGX Spark A/B test:** On a single DGX Spark, SGLang delivered **2.06× faster long-context prefill**, **+17.4% faster long-context decode**, and **~40 GiB lower unified memory usage** than vLLM—while passing chat, reasoning, tool calling, and vision out of the box.
> Looking for the vLLM recipe? See [inkling-2-mini-vllm-dgx-spark](https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark).

---

## Hardware & Model Overview

* **Hardware:** 1× NVIDIA DGX Spark (128 GB unified LPDDR5X, Blackwell native FP4)
* **Model:** `thinkingmachines/Inkling-2-Mini` — 26.4B total / 3.4B active (~5.4B effective across 3 looped middle-layer passes), multimodal (text + image in, text out), up to 1M context, bundled DSpark speculator
* **Precision:** Native mixed **NVFP4 + MXFP8** (~20 GB checkpoint, zero requantization)
* **Image:** `lmsysorg/sglang:inkling-2-mini` (`linux/arm64` native)

---

## Head-to-Head DGX Spark Benchmarks (SGLang vs. vLLM)

Measured on a single NVIDIA DGX Spark (`driver 580.178.04`, 121.6 GiB usable unified RAM) using the reproducible stdlib-only script in [`benchmarks/bench.py`](benchmarks/bench.py) over `/v1/chat/completions` with realistic coding/technical prompts:

| Metric | **SGLang + DSpark (this repo)** | vLLM + DSpark | Winner |
| :--- | :---: | :---: | :---: |
| **Smoke Tests** (Chat/Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | Tie |
| **`c=1` Short Decode (3-run avg)** | 49.56 tok/s *(peak 64.4)* | **52.30 tok/s** | vLLM (+5.5%) |
| **`c=1` Short TTFT** | 0.221s | **0.194s** | vLLM |
| **`c=1` 15.6K-Context Cold Prefill** | **4.48s** *(~3,480 tok/s)* | 9.23s *(~1,690 tok/s)* | **SGLang (2.06×)** |
| **`c=1` 15.6K-Context Warm Prefix TTFT** | 0.254s | **0.227s** | vLLM *(both ~0.2s)* |
| **`c=1` 15.6K-Context Decode** | **45.12 tok/s** | 38.43 tok/s | **SGLang (+17.4%)** |
| **`c=4` Concurrent Aggregate** | 87.88 tok/s *(peak 102.2)* | **91.17 tok/s** | vLLM (+3.7%) |
| **Unified RAM Used** | **65.2 GiB** *(56.4 GiB free)* | 105.5 GiB *(16.1 GiB free)* | **SGLang** |

Raw JSON outputs are included in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

### Option 1: One-command launch (`start.sh` / Docker Compose)

```bash
git clone https://github.com/thefullctx/inkling-2-mini-sglang-dgx-spark.git
cd inkling-2-mini-sglang-dgx-spark
cp .env.example .env
./start.sh
```

Follow startup logs (first run downloads the ~20 GB weights into `~/.cache/huggingface`):

```bash
docker logs -f inkling2-mini-sglang
```

Stop the server:

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

Once `/v1/models` is up on `http://127.0.0.1:8000`, run the included smoke test + benchmark suite (zero pip dependencies, Python 3 stdlib only):

```bash
python3 benchmarks/bench.py --label sglang-dspark
```

This tests:
1. Chat + `reasoning_content` extraction (`--reasoning-parser inkling`)
2. Function / tool calling (`--tool-call-parser inkling2`)
3. Multimodal vision input (base64 PNG)
4. `c=1` short decode, 15.6K cold/warm prefix TTFT & decode, and `c=4` concurrent throughput

---

## Tuning Notes

* `HF_HUB_DISABLE_XET=1` is enabled by default to avoid occasional Hugging Face Xet CAS decoding errors during initial download.
* `SGLANG_MEM_FRACTION=0.45` leaves ~56 GiB of unified memory free for the OS and other local agents; bump to `0.65`–`0.75` if you need heavier concurrency or ultra-long KV pools.
* Four SGLang speculative-decoding optimizations (`SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL`, `SGLANG_OPT_DSV41_DSPARK_MID_FOLD`, `SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE`, `SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS`) are pre-enabled for DGX Spark.
