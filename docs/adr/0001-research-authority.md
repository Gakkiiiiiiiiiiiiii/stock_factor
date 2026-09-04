# ADR 0001: Research and Paper Authority

Status: accepted

## Decision

Quant is the sole formal Paper Authority. Formal research consumes the
immutable Quant `market-snapshot.v1` and verified `content-factor-signal.v5.1`
references. Research artifacts are sealed first and carry their source,
checksum and OOS evidence; an exploratory or local-paper result is never
formal promotion evidence.

The repository has one configuration root, `config/`. A deprecated `configs/`
alias exists only in migration tooling, emits `DeprecationWarning`, and is
rejected by the formal loader. All loaded configuration is content-addressed
with canonical metadata and an environment binding.

## Consequences

Paper account/order/equity state remains upstream-owned. `experimental` CLI
commands are report-only and cannot authorize formal research or place orders.
Changes to cross-service contracts require an inventory checksum update and
compatibility review.
