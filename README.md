# Inkling-2-Mini on a Single NVIDIA DGX Spark (SGLang + DSpark + INT8 Draft + Verify Cap)

Tuned, lightweight single-node recipe for running **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** natively in **NVFP4 / MXFP8** with **DSpark speculative decoding** and **4 small read-only Python/Triton overlays** (zero C++/CUDA compilation) on a **single NVIDIA DGX Spark** using **SGLang**.

> **Highlights vs. Stock SGLang (1× DGX Spark, zero target weight quantization):**
> * **`c=1` short decode:** `45.2` → **50.4 tok/s** (`temp=0`, +11.4%) | **53.9 tok/s** (`temp=0.6`, +8.8% — beats stock vLLM!)
> * **15.6K long-context decode:** `40.0` → **51.4 tok/s (+28.6%)**
> * **15.6K warm prefix TTFT:** `0.27s` → **0.156s (156 ms, -42%)**
> * **`c=4` concurrent aggregate:** `87.6` → **101.8 tok/s (+16.2%)** *(104.0 tok/s with `VERIFY_CAP=4`)*
> * **Unified RAM headroom:** **~71.1 GiB used / ~50.5 GiB free** on 121.6 GiB Spark
> * **Accuracy:** Target weights & distribution unchanged (no lossy target quantization), reasoning + tools + vision all pass ✅.
>
> Looking for our vLLM recipe? See [inkling-2-mini-vllm-dgx-spark](https://github.com/thefullctx/inkling-2-mini-vllm-dgx-spark).

---

## Key Optimizations Over Stock SGLang

1. **Verify-Row Cap + Anchor-Expert Reuse (`overlay/inkling2_moe.py`, `overlay/dspark_verify.py`, `SGLANG_INKLING2_VERIFY_CAP=5`):**
   * Routes cut tail draft rows (`L..7`) to the anchor row's experts (`ids[:, L:, :] = ids[:, 0:1, :]`) inside the existing CUDA graph—slashing sparse MoE expert reads across all 48 layer passes with **0 GiB extra RAM**—while forwarding `cutoff_verify_lens=L` so neither greedy nor sampled acceptance ever reads cut rows.
   * Default `L=5` gives the best all-around balance; set `SGLANG_INKLING2_VERIFY_CAP=4` for peak `c=4` (`104.0 tok/s`), or `0` to disable.
2. **Custom Triton INT8 DSpark Draft Kernel (`overlay/gemv_int8.py`, `overlay/inkling2_mtp.py`, `SGLANG_INKLING2_DRAFT_INT8=1`):**
   * Quantizes **only** the draft-side LM head copy (`201,024 × 2,304`) and Markov fold (`201,024 × 256`, called 7×/step) to per-row symmetric INT8 at load time (**2.07×** faster Markov fold, **1.94–2.45×** faster draft LM head on GB10).
   * Target verifier stays 100% untouched in native BF16/NVFP4/MXFP8. *(Optional `SGLANG_INKLING2_DRAFT_LAYERS_INT8=1` also quantizes the 3 draft transformer layers—hits 106.6 tok/s at `c=4` `temp=0.6`, at the cost of +12 GiB RAM and slightly slower cold prefill, so kept off by default.)*
3. **`--page-size 16`** + **`SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1`** + **`SGLANG_OPT_MOE_QUANT_ONCE=1`** + **`--cuda-graph-max-bs-decode 8`**.

---

## Benchmarks (`1× NVIDIA DGX Spark`, All Smoke Tests ✅)

Measured with [`benchmarks/bench.py`](benchmarks/bench.py) over `/v1/chat/completions`:

| Metric | Stock SGLang (`page=128`) | Tuned (`page=16`) | + INT8 Draft | **+ Verify Cap `L=5` (Default)** | + Verify Cap `L=4` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Smoke Tests** (Reasoning, Tools, Vision) | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | **✅ / ✅ / ✅** | ✅ / ✅ / ✅ |
| **Greedy Match vs Stock** | Ref | Ref | 5/5 identical | **4/5 identical** *(1 near-tie `:` from GEMM batch rounding)* | 4/5 identical |
| **`c=1` Short Decode (`temp=0.0`)** | 45.22 tok/s | 45.16 tok/s | 47.97 tok/s | **50.37 tok/s** *(peak 57.0)* | **51.10 tok/s** |
| **`c=1` Short Decode (`temp=0.6`)** | 49.56 tok/s | — | 48.46 tok/s | **53.91 tok/s** | 51.95 tok/s |
| **15.6K Cold Prefill** | 4.55s | **4.47s** | 4.54s | **4.49s** | 4.50s |
| **15.6K Warm Prefix TTFT** | 0.27s | 0.21s | 0.20s | **0.156s (156 ms)** | **0.155s (155 ms)** |
| **15.6K Long-Context Decode (`temp=0.0`)** | 39.99 tok/s | 47.75 tok/s | 50.37 tok/s | **51.42 tok/s (+28.6%)** | 49.51 tok/s |
| **`c=4` Concurrent (`temp=0.0`)** | 87.63 tok/s | 90.44 tok/s | 94.87 tok/s | **101.81 tok/s** | **104.01 tok/s** |
| **`c=4` Concurrent (`temp=0.6`)** | 87.88 tok/s | — | 91.62 tok/s | 97.14 tok/s | **100.60 tok/s** |
| **Unified RAM Used** | 62.7 GiB | 62.8 GiB | 71.0 GiB | **71.1 GiB (~50.5 GiB free)** | 71.1 GiB |

Raw JSON logs for all 19 benchmark runs are in [`benchmarks/results/`](benchmarks/results/).

---

## Quick Start

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
* Hat tip to [sxuff/inkling-2-mini-spark](https://github.com/sxuff/inkling-2-mini-spark) for the INT8 DSpark draft-head idea and [knapcio/Inkling-2-Mini-1x-DGX-Spark](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark) for the verify-row cap + anchor-expert reuse insight — both implemented cleanly here in ~170 lines of read-only Python/Triton overlays for SGLang.
