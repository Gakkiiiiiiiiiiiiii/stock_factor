# stock_factor

Independent Quant Factor Research Service. It owns the factor DSL/VM, mining,
fitness, purged walk-forward/OOS validation, lifecycle persistence, alpha
scoring, mining worker, and paper-state contracts.

It consumes immutable Quant `market-snapshot.v1` market data and
`content-factor-signal.v5.1` signals strictly through HTTP
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

Paper APIs proxy the Quant Paper Authority (`trading.v1`) and preserve its
account/order/equity identities. The production composition is Quant-only and
fails closed for `local_experimental`; historical local state can be exported
only with the standalone read-only legacy exporter. Execution constraints such as
suspension, price limits, T+1 fills and transaction accounting remain owned
by the Quant service.

Quant is the only formal Paper Authority. `FACTOR_RUNTIME_PROFILE=staging` or
`prod` requires `FACTOR_PAPER_AUTHORITY=quant`, the `paper-account.v1`
contract, a non-empty Quant capability checksum, and `ALLOW_LOCAL_PAPER=false`.
No local Paper implementation is shipped in `src/stock_factor`. Formal research additionally
requires the `content-factor-signal.v5.1` contract and a registered content
capability checksum; all three content clocks and the content snapshot are
preserved in research lineage.

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
& D:\project\quant\.venv\Scripts\python.exe -m stock_factor.technical_transformer.training.train --config config\technical_v1.yaml
```

`config/technical_v1_local.yaml` is an engineering smoke configuration (64
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
  --gate-config config\technical_reliability_gate_v1.yaml `
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

## Configuration and CLI

`config/` is the sole configuration root. Configuration files carry an
explicit schema version, identity, environment and canonical checksum. The
offline CLI never contacts Quant or submits orders:

```powershell
python -m stock_factor.cli research config-inspect --path config/technical_v1.yaml
python -m stock_factor.cli research config-verify --path config/technical_v1.yaml
python -m stock_factor.cli artifact verify --path artifacts/research-artifact.json
python -m stock_factor.cli oos status --path artifacts/oos-status.json
```

Paper authority is Quant only for formal workflows. Local paper and other
experimental commands are explicitly non-formal and cannot create formal
promotion evidence. A temporary legacy path alias may emit a deprecation
warning, but formal configuration loading rejects legacy roots.
