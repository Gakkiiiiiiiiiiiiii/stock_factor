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

`configs/technical_v1_local.yaml` is the verified desktop run used for the
first checkpoint (64 symbols, 128-day windows, stride 5, CUDA FP16; stage
epochs 5/8/8/8 with 20 steps per epoch). The
unbounded `technical_v1.yaml` keeps the full-universe setting for the next
long run. Snapshot preparation joins quant PIT circulating capital to produce
`turnover = volume / circulating_capital`, preserves suspension rows with a
quality mask, and writes the qlib binary provider plus dataset manifest.

Inference on a saved checkpoint:

```powershell
& D:\project\quant\.venv\Scripts\python.exe -m stock_factor.technical_transformer.training.inference `
  --checkpoint artifacts\models\technical_local_v7\tech-v1-20260822T155151Z-wyckoff_phase_events-e008 `
  --dataset artifacts\datasets\technical_v1_local_v8 --split test --index 0
```
