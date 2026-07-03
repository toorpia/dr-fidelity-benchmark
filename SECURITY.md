# Security Policy

## Scope

This repository is a **research benchmark** — synthetic data, offline, CPU-only. It is a
reproducibility artifact, not a deployed service or a library meant to process untrusted input.

## Reporting a vulnerability

Please report security issues privately via GitHub (**Security ▸ Report a vulnerability**) rather than
a public issue. We will respond as soon as practical.

## Dependency pinning and Dependabot alerts

`requirements.txt` pins exact versions so the published metrics and figures reproduce bit-for-bit.
`torch==2.3.0` is pinned deliberately: PyMDE and PCC use torch as their optimizer backend, and the
validated results were produced with this version (cross-version floating-point differences can change
the exact metric values).

The `torch` advisories Dependabot flags are **not reachable** in this benchmark:

| advisory | affected API | used here? |
|---|---|---|
| CVE-2025-32434 (critical) | `torch.load(weights_only=True)` deserialization → RCE | **No** — we never call `torch.load`; no checkpoints are loaded. |
| CVE-2025-3730 / CVE-2025-2999 / CVE-2025-2998 (medium) | resource shutdown · `unpack_sequence` · `pad_packed_sequence` | **No** — RNN / packed-sequence APIs are not used. |
| CVE-2025-3001 / CVE-2025-2953 / CVE-2025-2149 / CVE-2025-2148 / CVE-2025-3000 (low) | `lstm_cell` · local DoS · quantized sigmoid · tuple handler · `jit.script` | **No** — these ops are not used. |

This project's only use of torch is `manual_seed`, `set_num_threads`, `as_tensor` (CPU), and PyMDE /
PCC's internal optimization over trusted, in-memory synthetic arrays. No untrusted data reaches any of
the affected functions, so these alerts are dismissed as **"vulnerable code not used."**

Reproducers who prefer a newer torch may install a patched release (`pip install "torch>=2.6.0"`);
expect minor numeric drift in the per-method results.
