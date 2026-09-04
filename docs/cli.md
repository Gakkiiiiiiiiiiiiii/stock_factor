# CLI

The `stock-factor` command is offline by default. It does not contact Quant,
create accounts, place orders, or mutate research state.

```powershell
stock-factor research config-inspect --path config/technical_v1.yaml
stock-factor research config-verify --path config/technical_v1.yaml --environment shared
stock-factor artifact verify --path artifacts/research-artifact.json
stock-factor oos status --path artifacts/oos-status.json
stock-factor experimental
```

`research config-verify` prints the source and canonical SHA-256 identity.
`artifact verify` reconstructs and verifies the immutable Artifact-first
payload. `experimental` is explicitly `formal_eligible: false`; it is not a
Paper Authority or a promotion path. The `admin` group is reserved for local,
non-mutating inspection utilities.
