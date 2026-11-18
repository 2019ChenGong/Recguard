# MailRecGuard: Defending LLM Agents against Persuasion-Based Memory Poisoning

This repository contains the code for the paper. It includes

- the two attacks (SFP and TPB) and the forwarded-message scenario (FWD),
- eight memory frameworks behind one interface,
- seven existing defenses and the defense proposed in the paper (MailRecGuard),
- the scripts that run the experiments and build the tables.

All experiments use Qwen3.8-27B served locally with vLLM. No external API is needed.

## 1. Repository layout

| Path | Content |
|---|---|
| `scripts/exp_backends_qwen.py` | Experiment driver: builds the inbox, applies a defense, lets a memory framework ingest the inbox, asks the recommendation queries, records the results. MailRecGuard is `_typed_ledger_transform_emails`. |
| `memory_evidence.py` | Binding checks and canonical records of MailRecGuard (code only, no model). |
| `scripts/pilot_qwen38_evidence.py` | Prompts of the claim extractor and the stance checker (`PROMPTS`). |
| `qwen_shopping_experiment.py` | Writer agent, recommender, and the baselines Memory Sanitization, StateGuard, Trojan Hippo. |
| `scripts/consumption_defenses.py` | Baselines A-MemGuard and SMSR. |
| `scripts/mage_shadow_memory.py` | Baseline ShadowMem. |
| `memory_influence_guard.py` | Counterfactual guard. |
| `memory_authority.py`, `shopping_experiment.py`, `agent_interface.py` | Shared helpers. |
| `shopping_data.py`, `shopping_data_v2.py`, `daily_inbox_data.py` | Users, product catalog, genuine notes, attack messages, everyday messages. All data is synthetic. |
| `scripts/letta_memory_adapter.py`, `scripts/local_embedding_server.py` | Client and embedding server for the Letta framework. |
| `scripts/openai_sandbox_memory_adapter.py` | Adapter for the sandbox memory of the OpenAI Agents SDK (OpenAI-SM). |
| `scripts/qwen_native_api.py` | Transport adapter between Qwen-Agent and the vLLM server. |
| `scripts/run_grid.sh` | Runs one grid of (attack, defense) pairs on the selected frameworks. |
| `scripts/run_paper_experiments.sh` | Runs the experiments of the paper, group by group. |
| `scripts/start_vllm_qwen.sh`, `scripts/start_vllm_shadowmem_judge.sh` | Start the model server and the ShadowMem judge. |
| `scripts/natural_tables.py`, `scripts/noise_table.py`, `scripts/natural_decisions.py`, `scripts/typed_ledger_trace_table.py` | Build the tables from the logs. |
| `scripts/probe_sanitization_natural.py`, `scripts/probe_stateguard_natural.py` | Count the decisions of Memory Sanitization and StateGuard per message kind. |
| `configs/` | Example endpoint files. |
| `tests/` | Unit tests. |

## 2. Requirements

**Hardware.** One GPU with 48 GB of memory (or two GPUs with 24 GB each) for the model
server. The ShadowMem baseline needs one more GPU for its judge model.

**Software.** Linux, Python 3.12. Three environments are used:

```bash
# (a) experiment environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (b) model server (separate environment)
pip install vllm==0.25.1

# (c) only for the Letta framework (separate environment), see Section 7
pip install "letta==0.16.8" asyncpg pg8000 pgvector
```

## 3. Start the model server

```bash
# one 48 GB GPU
CUDA_VISIBLE_DEVICES=0 scripts/start_vllm_qwen.sh
# two 24 GB GPUs
CUDA_VISIBLE_DEVICES=0,1 VLLM_TENSOR_PARALLEL_SIZE=2 VLLM_GPU_MEMORY_UTILIZATION=0.92 scripts/start_vllm_qwen.sh
```

The script serves `Qwen/Qwen3.8-27B-FP8` at a pinned revision with a 32K context on port 8000.
Then tell the clients where the server is:

```bash
cp configs/vllm.env.example configs/vllm.env     # edit host and port if the server is remote
```

## 4. Quick start

```bash
source .venv/bin/activate
scripts/run_paper_experiments.sh smoke           # one user, two categories
python scripts/natural_tables.py --daily 20 --categories 2 --n-full 2
```

The first command runs the SFP attack on the Explicit and RAG frameworks, once without a
defense and once with MailRecGuard. It takes about 25 minutes when the server is shared with
other runs, and less on an idle server. The second command prints the ASR and HM of both
runs. Expected result: without a defense the ASR is 100 % and the HM is 0 %; with
MailRecGuard the ASR is 0 % and the HM is 100 %.

## 5. Running the experiments of the paper

Every experiment is a call of `scripts/run_grid.sh` with environment variables.
`scripts/run_paper_experiments.sh <group>` issues the calls of one group:

| Group | Experiment | Calls |
|---|---|---|
| `existing` | No defense and the existing defenses under SFP and TPB, plus the clean inbox | 54 |
| `shadowmem` | ShadowMem under SFP and TPB (start the judge first, Section 7) | 12 |
| `label` | Source label stored in memory, source filter before memory, forwarded messages | 30 |
| `ours` | MailRecGuard under SFP, TPB, FWD, and on a clean inbox | 12 |
| `ablation` | MailRecGuard without one component; K = 1 and K = 3 | 54 |
| `labelerror` | 5 %, 10 %, 20 % wrong source labels, three error patterns | 144 |
| `count` | 1, 5, 10 SFP messages and 1, 3, 10 TPB messages per category | 30 |
| `size` | 0, 100, 1000 everyday messages per user | 30 |
| `dormancy` | All attack messages arrive before 100, 300, 1000 everyday messages | 30 |

```bash
DRY_RUN=1 scripts/run_paper_experiments.sh existing        # print the calls
scripts/run_paper_experiments.sh ours                      # run them one after another
FRAMEWORKS="local" scripts/run_paper_experiments.sh ours   # skip Letta and OpenAI-SM
```

The calls are independent. To use several model servers, print the calls with `DRY_RUN=1`
and start them in parallel with a different `QWEN_ENV_FILE` each.

A single call looks like this:

```bash
STYLES="spoofed_first_party" DEFENSES="typed_ledger+ownpart+conflict" \
BACKENDS="explicit context rag" TAG=example scripts/run_grid.sh
```

**Setting.** Three users, ten queried product categories, 30 queries per cell. Each inbox
holds 300 everyday messages, 20 genuine preference notes (one per category) and the attack
messages (SFP: 3 per category, TPB: 5 per category), in random order.

**Run time.** Measured with the server shared by several runs. A full inbox takes about
2 to 4 hours for Explicit, Context and RAG together, 3 hours per user for Mem0, 40 minutes
for MemoryBank, 30 minutes for TencentDB, 6 to 10 hours for Letta and 5 to 8 hours per user
for OpenAI-SM. With the source filter or MailRecGuard only a few records reach the
framework, and the six local frameworks take about 40 minutes together.

## 6. Results and tables

Each call writes to `logs/`:

- `backends_grid_raw_<run>.json`: one row per query. `is_target_top1` means that the agent
  recommends the attacker's product. `is_preference_top1` means that it recommends the
  product that the user prefers.
- `backends_grid_summary_<run>.json`: the settings of the run and its ASR and HM per cell.
- `typed_ledger_trace_<run>.json`: for MailRecGuard, the step at which each message stops.

ASR is the share of queries with `is_target_top1`. HM is the share with `is_preference_top1`.

```bash
python scripts/natural_tables.py [--latex]         # ASR / HM per framework, defense and attack
python scripts/natural_tables.py --daily 100       # a run of the size sweep
python scripts/natural_tables.py --order attack_first --daily 300   # a run of the dormancy sweep
python scripts/noise_table.py                      # wrong source labels
python scripts/typed_ledger_trace_table.py --defense typed_ledger+ownpart+conflict
python scripts/natural_decisions.py [--latex]      # share of attack / genuine items stopped by each defense
```

`natural_decisions.py` also reads the output of the two probe scripts:

```bash
source configs/vllm.env
export QWEN_CATALOG_V2=1 QWEN_HM_EMAILS_PER_CATEGORY=1 QWEN_SPOOF_COUNT=3 PYTHONNOUSERSITE=1
for s in spoofed_first_party legit_marketing; do QWEN_EMAIL_STYLE=$s python scripts/probe_sanitization_natural.py; done
for u in p01 p02 p03; do QWEN_SHARE_WRITER_NOTES=1 python scripts/probe_stateguard_natural.py $u; done
```

The tables are written to `logs/tables/`.

## 7. Frameworks and baselines that need an extra service

**Letta.** Letta runs as a server with PostgreSQL (with the pgvector extension) and an
embedding endpoint. Start them in this order on the machine that runs the Letta experiments:

```bash
# 1. PostgreSQL with pgvector, database "letta", user "letta"
# 2. embedding server (experiment environment)
python scripts/local_embedding_server.py --port 8285 --model sentence-transformers/all-MiniLM-L6-v2
# 3. Letta server (Letta environment); it uses the vLLM server as its model provider
source configs/vllm.env
LETTA_PG_URI=postgresql+pg8000://letta@127.0.0.1:5432/letta \
OPENAI_API_KEY=$VLLM_API_KEY OPENAI_BASE_URL=$VLLM_API_BASE letta server --port 8283
```

`letta==0.16.8` does not ship its migration scripts. Create the tables once with
`Base.metadata.create_all` of the Letta ORM, then run
`ALTER TABLE messages ALTER COLUMN sequence_id SET DEFAULT nextval('message_seq_id_seq');`.
The client reads `LETTA_BASE_URL` (default `http://127.0.0.1:8283`) and `LETTA_EMBED_ENDPOINT`
(default `http://127.0.0.1:8285/v1`).

**OpenAI-SM.** The sandbox memory of the OpenAI Agents SDK runs inside the experiment
environment and uses the vLLM server. It needs the 32K context of the server.

**ShadowMem.** Start the judge released by its authors on a second GPU:

```bash
CUDA_VISIBLE_DEVICES=1 scripts/start_vllm_shadowmem_judge.sh
cp configs/shadowmem_judge.env.example configs/shadowmem_judge.env
```

## 8. Names in the paper and in the code

| Paper | Code |
|---|---|
| SFP, TPB, FWD | `STYLES=spoofed_first_party`, `legit_marketing`, `forwarded_ad` |
| Explicit, Context, RAG, Mem0, MemoryBank, TencentDB, Letta, OpenAI-SM | `BACKENDS=explicit context rag mem0 memorybank tencentdb letta openai_sandbox` |
| Inbox with the attack / clean inbox / no email | `CONDITIONS=poisoned_neutral_daily_inbox` / `clean_preference_neutral_daily_inbox` / `baseline_no_email` |
| No defense | `DEFENSES=none` |
| Memory Sanitization | `DEFENSES=content_classifier` |
| StateGuard | `DEFENSES=memory_stateguard` |
| Trojan Hippo | `DEFENSES=limit_memory_length` |
| A-MemGuard, SMSR, PPMF, counterfactual guard | `DEFENSES=none` with `QWEN_QUERY_VARIANTS=amemguard,smsr,action_gate,strict_guard` (one ingestion, one query pass per variant) |
| ShadowMem | `DEFENSES=mage` |
| Source label stored in memory | `DEFENSES=channel_stamp` |
| Source filter before memory | `DEFENSES=channel_taint` |
| **MailRecGuard** | `DEFENSES=typed_ledger+ownpart+conflict` |
| without the source check (Step 1) | `typed_ledger+ownpart+conflict+noauth` |
| without the own-text check (Step 2) | `typed_ledger+conflict` |
| without the stance check (Step 2) | `typed_ledger+ownpart+conflict+nostance` |
| without conflict confirmation (Step 4) | `typed_ledger+ownpart` |
| the user does not answer a confirmation | `typed_ledger+ownpart+conflictna` |
| K claims per message | `...+k1`, `...+k3` (default 2) |
| Share of wrong source labels | `QWEN_CHANNEL_NOISE` (false accepts), `QWEN_CHANNEL_NOISE_FR` (false rejects, default: same rate), `QWEN_CHANNEL_NOISE_SEED` (error pattern) |
| Attack messages per category | `QWEN_SPOOF_COUNT` (SFP), `QWEN_TPB_COUNT` (TPB) |
| Everyday messages per user | `QWEN_DAILY_BENIGN_COUNT` |
| Attack messages arrive first | `QWEN_DAILY_ORDER=attack_first` |

## 9. Notes on the implementation

- The source label of a message comes from the dataset: it says whether the message was
  sent from the user's own account. Wrong labels are simulated with `QWEN_CHANNEL_NOISE`.
- Each message of the dataset carries its product category. The claim extractor reads the
  catalog of that category. Messages without a product category are skipped.
- An inbox is processed as one batch. The answer of the user to a confirmation request is
  simulated with the true preference of the user.
- The registry of the driver contains more frameworks than the eight used in the paper.
  They are not needed and not supported here.
- Mem0 rewrites and merges its entries with a language model, and its results vary between
  runs of the same inbox.

## 10. Unit tests

```bash
python -m unittest tests.test_memory_evidence tests.test_memory_authority tests.test_memory_influence_guard
python -m unittest discover -s tests          # all tests; needs the experiment environment
```
