# Research readiness and SLOs

`/health/live` is a liveness signal only. Research, ML, and Paper readiness are
independent reports and must not be inferred from liveness.

Formal research admission requires fresh Quant `market-snapshot.v1`, Content
`content-factor-signal.v5.1`, verified contract checksums, reachable append-only
artifact storage, an OOS lease-capable repository, and resource headroom of at
least 512 MB, 30 seconds of deadline, and queue depth no greater than 100.

Readiness evidence is frozen with a UTC timestamp and SHA-256 hash. It is
revalidated immediately before OOS authorization. Stable blocking codes are
used for alerts; identifiers, symbols, exception text, and artifact hashes are
not metric labels.

Initial operational targets: readiness checks complete within 2 seconds; 99% of
formal admission checks pass within 5 seconds; dataset/mining/OOS/artifact
latency and error counters use the allowlisted metrics in
`stock_factor.observability.metrics`. Alert on `research_readiness=0`, stale
market/content freshness, queue depth over 100, or OOS lease capability loss.
The starter dashboard and alert thresholds are versioned in
`config/monitoring/research_slo.yaml`.
