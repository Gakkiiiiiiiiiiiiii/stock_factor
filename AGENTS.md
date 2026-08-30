# AGENTS.md

## Repository Ownership

本仓库负责独立的量化因子研究服务：factor DSL/VM、因子挖掘、fitness、purged
walk-forward/OOS 验证、生命周期持久化、alpha scoring、technical transformer、
mining worker 以及 paper-state contract。

本仓库不负责 `stock_agent`、`stock_content` 或 `quant` 的内部实现。跨仓库集成
只能通过 `contracts/` 定义的兼容契约、`src/stock_factor/adapters/http/` 中的
HTTP provider/client 或消息接口完成，不得直接 import 其他仓库的实现。

## Architecture Boundaries

- `src/stock_factor/domain/` 负责纯领域模型和版本化研究对象。
- `src/stock_factor/application/` 负责挖掘、因子集、OOS 和 paper use cases。
- `src/stock_factor/engine/` 负责确定性 DSL/VM、fitness、研究切分、统计验证、生命周期和 promotion gate。
- `src/stock_factor/technical_transformer/` 负责技术特征、训练、评估、可靠性 gate 和模型 promotion。
- `src/stock_factor/ports/` 负责对外 provider、交易日历和符号等抽象。
- `src/stock_factor/adapters/http/` 是访问 `quant`、`stock_content` 等外部服务的唯一 HTTP 边界。
- `src/stock_factor/adapters/postgres/` 负责持久化实现；schema 变化必须有编号 migration 和测试。
- `src/stock_factor/api/` 负责 HTTP entry points；`src/stock_factor/workers/` 负责异步 job 执行。
- `contracts/` 的 public contract 变化必须有兼容性/回归覆盖和必要的版本说明。
- 保持 local/remote provider 的回滚路径和 snapshot/OOS sealing 语义，不为了局部任务删除既有兼容路径。

## Research and Trading Safety

- 默认只运行 deterministic fixture、unit/contract test、backtest、replay、paper 或 shadow 流程。
- 不得绕过 timestamp cutoff、future-leak、purged split、OOS authorization、one-shot consumption、multiple-testing、replay determinism 或 immutable snapshot 约束来通过测试。
- `SMOKE` 评估结果不能直接晋升为 `ACTIVE`；模型/因子 promotion 必须有完整 reliability/OOS gate 证据。
- 未得到用户对具体动作的明确授权，不得向外部账户、broker 或 QMT 提交真实订单，不得启用 LIVE 交易或修改账户/执行开关。
- 涉及 paper 账户、持仓、订单、权益快照或外部交易状态时，先报告风险和所需证据，不要自行扩大操作范围。

## Critical Paths

- `src/stock_factor/domain/`、`application/`、`engine/`：因子研究核心路径。
- `src/stock_factor/technical_transformer/`：技术模型数据、训练、评估和 promotion 路径。
- `src/stock_factor/adapters/http/`、`ports/`、`contracts/`：跨服务和 public boundary。
- `src/stock_factor/adapters/postgres/`、`migrations/`：持久化、schema 和研究证据。
- `src/stock_factor/api/`、`workers/`：服务 entry points 和异步任务。
- `tests/`：领域、API、contract、architecture、研究完整性、OOS、replay、paper 和技术 transformer 回归测试。
- `configs/`、`scripts/`：研究配置、快照准备、数据校验和评估入口。

## Build and Test Matrix

安装测试依赖：

    python -m pip install -e ".[test]"

快速确定性测试：

    python -m pytest -q --ignore=tests/test_integration.py

Lint：

    python -m ruff check src tests

Architecture/contract gate：

    python -m pytest -q tests/test_alpha_score_contract.py tests/test_api.py tests/test_remote_market_contract.py tests/test_architecture.py tests/test_architecture_boundaries.py tests/test_paper_runtime_authority.py

Research integrity gate：

    python -m pytest -q tests/test_timestamp_cutoff.py tests/test_final_oos_seal.py tests/test_research_v2_authority.py tests/test_oos_audit.py tests/test_oos_concurrent_consumption.py tests/test_multiple_testing.py tests/test_dsr.py tests/test_pbo.py tests/test_metric_scope.py tests/test_research_dataset_refs.py tests/test_statistical_experiment.py tests/test_fitness_validation.py

技术 transformer 或数据快照任务应额外运行其对应的 `tests/technical_transformer/`、snapshot、causality、leakage 和 evaluation 测试。Postgres、外部 HTTP、重型 ML、Docker 和集成测试是环境门禁；依赖不可用时记录为环境失败，不通过业务代码绕过。

完整本地相关 suite：

    python -m pytest -q --ignore=tests/test_integration.py

## Generated and Large Paths

除非任务明确要求，不要读取或修改：

- `artifacts/`、模型 checkpoint、训练数据集、量化行情、缓存、日志和大型报告；
- `.venv/`、`.pytest_cache/`、`.pytest-*`、`.test-*`、coverage 文件和临时 workspace；
- `*.db`、`*.sqlite*`、运行时 JSONL/CSV 以及 `storage/` 等生成数据；
- `.env` 及任何 API key、cookie、账户或 broker 状态文件。

优先读取 fixture、contract、manifest、JSON/Markdown 摘要和最小测试输出，不要把完整长日志放入主 agent 上下文。

## Change Rules

- 做最小、可解释的 patch，不做无关重构或全仓格式化。
- 不新增生产依赖，除非任务明确允许。
- bug fix 应在可行时增加 regression test。
- 不删除、跳过、弱化或改写测试来让实现通过。
- 修改 public contract、数据库 schema、研究切分、评估 gate 或 promotion 语义时，必须同步更新 contract/migration、测试和文档。
- 普通 feature 任务不得修改 `AGENTS.md`、`.codex/` 配置或用户级 Codex 配置。
- 不提交模型、数据库、数据集、凭据或与任务无关的已有工作区改动。

## Code Review Rules

按以下顺序审查：

- correctness、失败处理和确定性；
- future-leak、timestamp cutoff、OOS 授权、统计有效性和 replay；
- snapshot/manifest 不可变性、数据血缘、幂等和研究证据完整性；
- contract、向后兼容性和跨服务边界；
- paper/live、promotion 和外部账户安全；
- 测试是否覆盖 acceptance criteria；
- 是否出现无关文件、生成产物或隐式依赖。

P0/P1 findings 必须修复；P2 原则上修复或明确记录原因；P3 纯格式问题不阻断验收。

## Definition of Done

仓库级 feature 只有在以下条件都有证据时才能宣布完成：

- 每条 acceptance criterion 都有具体证据；
- targeted tests 和 full relevant deterministic suite 通过；
- 适用的 architecture、contract、research-integrity、OOS、replay、golden/data-quality gate 通过；
- Ruff 通过，或明确记录不适用；
- 最终 diff 无无关改动、无未解释 contract/schema/config drift；
- 无未解决的 P0/P1 review finding；
- 没有未经授权的真实交易、promotion 或账户操作；
- baseline 结果和既有失败已记录。
