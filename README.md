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
