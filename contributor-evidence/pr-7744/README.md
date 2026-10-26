# FlagGems #7744: measured CUDA comparison evidence

Measured on 2026-10-27 (Asia/Shanghai). PR production source is unchanged by this evidence branch.

- Candidate: `8ded041abac535db3cb04ca350ebc5c903678e4a`.
- Controlled baseline: the same candidate checkout and regression tests, with **only** `src/flag_gems/ops/eq.py` restored from `415ea38c63141382b0051eac5ef0fa22cbbbec69`. That source contains the two unconditional FP32 casts. This isolates the production patch; it is not a full older-tree build.
- GPU: NVIDIA GeForce GTX 1650, 4 GB, compute capability 7.5, driver 551.86.
- Runtime: Windows x64, Python 3.12.10, PyTorch 2.10.0+cu126, triton-windows 3.2.0.post21 (Triton reports 3.2.0). Both versions run the real CUDA kernels. An int64 Triton addition smoke test also passed.
- The complete `flag_gems` package imports normally. No operator, dispatcher, registration, promotion or test source was patched for Windows. `host-shims/fcntl.py` supplies a Windows advisory lock for the upstream pytest JSON report writer, which imports POSIX fcntl. The shim is outside the source tree and is identical in both variants.

## Results

| Check | Controlled baseline | Candidate |
|---|---:|---:|
| Six introduced int32/int64 precision tests | 0 passed / 6 failed | **6 passed / 0 failed** |
| All three requested files, 60 cases | 36 passed / 24 failed | 54 passed / 6 failed |
| Non-BF16 subset (6 new + 36 existing) | 36 passed / 6 failed | **42 passed / 0 failed** |
| Extended CUDA-oracle comparison matrix, 326 cases | 285 passed / 41 failed | 288 passed / 38 failed |
| Newly failing matrix cases relative to baseline | — | **0** |

The candidate's remaining six failures are exactly `test_eq_scalar[dtype2-shape0]` through `test_eq_scalar[dtype2-shape5]`: the dtype is BF16 and ptxas reports that `.bf16` / `cvt.f32.bf16` requires `sm_80` or higher. The baseline also fails those cases, plus the other twelve BF16 cases and all six new integer regressions. The final clean run explicitly deselects **all 18 BF16 cases**, including the twelve that happen to pass on the candidate; this does not claim general BF16 support on Turing.

The matrix covers all 64 ordered pairs of bool/int8/int16/int32/int64/float16/float32/float64, contiguous and strided broadcast inputs, twelve Python scalar values, and six integer precision boundary checks. All 128 tensor/tensor matrix cases pass in both variants. The three additional candidate improvements are integer scalar comparisons around 2**24 (int32 and int64) and 2**53 (int64). The 38 remaining scalar mismatches are present on **both** variants; their exact case IDs are retained in `summary.json`. They include bool/integer scalar call sequences and scalars outside narrow integer ranges. Their root causes require separate investigation; this evidence does not establish that every scalar edge case is correct.

## Reproduce

Check out the candidate source separately from this evidence directory. Use the pinned runtime above; Triton 3.3+ drops this GPU architecture, and PyTorch 2.6 cannot import this FlagGems revision because an unrelated operator references a newer dtype. PyTorch 2.10 + Triton 3.2 is a measured local configuration, not a claim of upstream certification for that version pair.

In PowerShell, with `$source` set to the candidate checkout, `$proof` to this directory, and the isolated environment active:

```powershell
$env:PYTHONPATH = "$proof\host-shims;$source\src"
Set-Location $source
python "$proof\cuda_smoke.py"
python -m pytest -q tests/test_eq_integer_precision.py tests/test_eq.py tests/test_equal.py
python -m pytest -q tests/test_eq_integer_precision.py tests/test_eq.py tests/test_equal.py -k 'not dtype2'
python "$proof\promotion_matrix.py"
git fetch --depth 1 origin 415ea38c63141382b0051eac5ef0fa22cbbbec69
python "$proof\run_baseline.py" --repo $source
git diff --exit-code
```

The full candidate pytest command exits 1 for the six hardware-limited BF16 scalar cases. The non-BF16 command exits 0 (42 passed, 18 deselected). The matrix exits 1 because it also exposes existing scalar mismatches. The baseline runner records these expected nonzero child exits, and restores the candidate source in `finally`; `baseline-control.json` records the restoration and source SHA256. See `summary.json` for the per-case target results and matrix failure sets.

## Review limits

- Run the complete three-file suite on an Ampere-or-newer GPU with the project's Linux/FlagTree stack to cover BF16 and current compiler integration.
- Upstream CI and maintainer review remain required. Ready for review is not a merge-readiness or all-backend support claim.
- No performance claim is made; a matched benchmark was not run.
