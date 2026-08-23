# stock_factor

Independent Quant Factor Research Service. It owns the factor DSL/VM, mining,
fitness, purged walk-forward/OOS validation, lifecycle persistence, alpha
scoring, mining worker, and paper-state contracts.

It consumes market data and `content-factor-signal.v1` strictly through HTTP
ports and does not import `stock_agent` or `stock_content` implementations.

The market adapter expects the remediation contract `POST /v1/bars/batch` with
`symbols`, `dates`, column-oriented `bars`, `data_version`, and
`data_snapshot_id` in the response. Deployments must provide that endpoint (or
inject another `MarketDataProvider`) before starting mining workers.

## Implemented vertical slice

```text
POST mining job -> PostgreSQL -> leased worker
  -> MarketDataProvider + ContentSignalProvider
  -> feature panel -> Factor VM -> IC/RankIC/ICIR/TopK
  -> purged walk-forward -> factor library
  -> list/get/evaluate/alpha APIs
```

Paper APIs persist frozen T-1 target orders, account state and equity snapshots.
Execution constraints such as suspension, price limits, T+1 fills and
transaction accounting remain the next paper-trading migration increment.

## Run

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The API listens on `http://localhost:8200`; `stock-factor-worker` runs as a
separate process. Production should apply SQL in `migrations/` before startup.

## Technical Transformer V1

The V1 training pipeline lives under `src/stock_factor/technical_transformer`.
It consumes only a verified, immutable quant snapshot; it does not read a
mutable `latest.parquet` directly.

```powershell
$env:PYTHONPATH = "D:\project\stock_factor\src"
& D:\project\quant\.venv\Scripts\python.exe scripts\prepare_technical_snapshot.py --quant-root D:\project\quant
& D:\project\quant\.venv\Scripts\python.exe -m stock_factor.technical_transformer.training.train --config configs\technical_v1.yaml
```

`configs/technical_v1_local.yaml` is an engineering smoke configuration (64
symbols, 128-day windows, stride 5, CUDA FP16). It uses the V2 feature/label
schemas, a deterministic 20% instrument holdout, structured masking, gradient
accumulation, and separate encoder/head learning rates. The unbounded
`technical_v1.yaml` is the research-scale configuration; the local run must
not be treated as a reliable model. Snapshot preparation joins quant PIT
circulating capital to produce `turnover = volume / circulating_capital`,
preserves missing-turnover calendar rows with observation flags, and writes
the qlib-compatible provider plus V2 dataset manifest.

Inference on a saved checkpoint:

```powershell
& D:\project\quant\.venv\Scripts\python.exe -m stock_factor.technical_transformer.training.inference `
  --checkpoint artifacts\models\technical_local_v8\<checkpoint> `
  --dataset artifacts\datasets\technical_v2_local --split time_test --index 0
```

After training, a checkpoint is only a candidate. Run the frozen evaluator to
produce JSON and Markdown evidence; only a checkpoint with a PASS Reliability
Gate may be promoted to ACTIVE:

```powershell
& D:\project\quant\.venv\Scripts\python.exe -m stock_factor.technical_transformer.evaluation.run `
  --mode PRODUCTION `
  --checkpoint artifacts\models\technical_local_v8\<checkpoint> `
  --dataset artifacts\datasets\technical_v2_local `
  --gold-set artifacts\gold_sets\wyckoff_v1 `
  --baseline-root artifacts\baselines\technical\<checkpoint> `
  --gate-config configs\technical_reliability_gate_v1.yaml `
  --report artifacts\reports\technical\<checkpoint>
```

The orchestrator executes frozen split metrics, causality, Gold Set inference,
fixed-split embedding probes, invariance/occlusion checks and GRU/MLP
baselines before writing a V2 report. `SMOKE` is never eligible for ACTIVE.
Promotion is a separate lifecycle action:

```powershell
& D:\project\quant\.venv\Scripts\python.exe -m stock_factor.technical_transformer.evaluation.promote `
  --checkpoint artifacts\models\technical_local_v8\<checkpoint> `
  --report artifacts\reports\technical\<checkpoint>\reliability_report.json `
  --target ACTIVE
```
