# agents-mcp

[Русский](README.md) | English

An MCP server for a platform of specialized LLM agents. Each agent is a folder
`agents/<name>/` with `prompt.md` and `config.toml`; adding a new agent does not
require rebuilding the service. The repository contains only the engine itself:
you write the agents yourself (the format is described in ["Adding a New Agent"](#adding-a-new-agent)).

## Features

- **MCP tools**: agent invocation — `invoke_agent`, `agent_run` (starts in the
  background without waiting), `wait_agent`, `agent_cancel`, `chain_cancel`
  (stops the whole chain of work for a task); information —
  `list_agents`, `get_agent_info`, `agent_history`, `health`; service management —
  `config_reload`, `prepare_shutdown`; task board — `task_create`, `task_get`,
  `task_set_status`, `artifact_write`, `artifact_read`, `event_append`;
  file tools — `fs_read_file`, `fs_write_file`, `fs_edit_file`, `fs_mkdir`,
  `fs_list_dir`; skills — `skill_load`.
- **Invocation without waiting** — `agent_run`: starting returns a `call_id` and
  result file path immediately, and checking by `call_id` responds instantly.
  Neither mode holds the caller, so the client does not need an intermediary script.
- **Background call cancellation** — `agent_cancel`: the call row is closed with
  an error, the result file is completed, and a waiter does not hang.
- **Chain stop** — `chain_cancel(task_id)`: cancels every live background call of
  the task and marks the task `cancelled` (the reply lists the cancelled
  `call_id`s; a closed task answers `already_closed`, an unknown one `not_found`).
  This is how `scripts/code_chain.py` is stopped: the script itself sees the
  `cancelled` status, writes `result.json` with status `cancelled` and exits with
  code 4 — killing Windows processes is no longer needed.
- **Model providers**: direct HTTP connections through
  `[providers.direct.<name>]` can be OpenAI-compatible (`api = "openai"`, the
  default value) or Anthropic Messages (`api = "anthropic"`, including MCP
  tools); Claude through Claude Code CLI (`claude -p`, with its own tools and
  MCP); OpenAI through Codex CLI. Plus `mock` for tests.
- **Two ways to communicate with the client**: Streamable HTTP (the default,
  without session state — no "404 Session not found" after a restart) and
  standard input/output (`--transport stdio`) for clients that do not support
  HTTP. The toolset is identical.
- **Tools for an agent — from both kinds of MCP servers**: both those available
  over HTTP and those launched as a process (`command` in `mcp_config`).
- **Reload without restart**: the agents directory is picked up by a watcher
  (500 ms delay), and the main configuration is picked up by a watcher or the
  `config_reload` tool.
- **Response cache** in storage (sha256 key, lifetime configured per agent).
- **Call history** in storage — metrics, tracing, service instance.
- **Service log** — a separate SQLite file `<log_dir>/agents-mcp.logs.db`
  (`events` table, event fields compressed with zstd), retained for 30 days. It
  works even when the primary database is unavailable. There are no text logs.
  `RUST_LOG` sets the overall level, but noisy libraries (`hyper`, `h2`, `axum`,
  `tower_http`, `tower`, `rmcp`, `tokio_util`) remain at `warn` until explicitly
  named in `RUST_LOG` (for example, `rmcp=debug`).
- **Choice of storage**: embedded SQLite by default — the service starts without
  an external database and creates the file itself; PostgreSQL — at the specified
  address (see ["Storage"](#storage)).
- **One instance per `log_dir`**: a PID lock with process name verification.
- **Graceful shutdown** on Ctrl+C/Break/Close/Shutdown (Windows) and SIGTERM
  (Unix); before stopping, use `prepare_shutdown` so running calls are not cut off.

## Storage

All primary service state (task board, call history, run turns, response cache)
resides in a single storage backend. The service log is written separately to
`<log_dir>/agents-mcp.logs.db`, so a failure of the primary database does not
lose it. By default, this is **embedded SQLite**: no external database is needed,
and the database file and schema are created automatically on first start. If a
PostgreSQL address is set in `[storage].task_store_dsn` (preferably through the
`AGENTS_MCP_TASK_STORE_DSN` environment variable, which overrides the
configuration), the service uses it.

| Address | What is selected |
|---|---|
| not set or empty | embedded SQLite at `[storage].sqlite_path` |
| `sqlite://path` | embedded SQLite at the specified path |
| `postgres://…`, `postgresql://…`, or `host=… user=…` | PostgreSQL |
| any other scheme | startup error; the error text contains only the scheme, and the password from the address is not included |

**SQLite** requires no configuration: the schema is applied on every open,
repeated starts are safe, and the journal mode is WAL.

**PostgreSQL**: the service does not create the schema itself — apply the files
in `migrations_pg/` in order (001–006); they are idempotent. Keep the password in
an environment variable, not in the configuration. Pool size is
`[storage].task_store_pool` (8 by default). On startup, the service closes with
an error only its own stuck calls — by instance in `agent_calls.instance`
(`[server] instance`, "machine name:port" by default), so multiple services using
one database must have distinct instances. On an already running database, apply
the missing migrations before installing the new build (`005` — service
instance, `006` — result file path): without these columns, writing a call
will fail.

A custom database is connected by implementing the `Store` trait
(`src/store/mod.rs`) and adding an address-parsing branch in
`parse_backend`/`connect`; see `src/store/sqlite.rs` for an example.

## Machine Layout (Example)

```
C:\agents-mcp\
├── bin\agents-mcp.exe          # binary
├── configs\agents-mcp.toml     # passed with --config or the AGENTS_MCP_CONFIG variable
├── agents\                     # your agents (picked up at runtime)
├── data\agents-mcp.sqlite      # embedded SQLite: board, calls, cache
├── logs\agents-mcp.logs.db     # service log (events), separate from the primary database
├── logs\agents-mcp.pid         # PID lock
└── runs\                       # result files of background agent_run calls
```

Relative paths in the configuration (`sqlite_path`, `log_dir`, `runs_dir`,
`agents_dir`) are resolved from the directory containing the configuration file
itself — hence the `../` in the example.

## Confidentiality

Starting an agent sends data to a third-party model provider. There is no
isolation by default, so whoever deploys the service chooses the operating mode.

The provider receives:

- the task text (the `input` field) and the agent's complete system prompt;
- the contents of files read by file tools;
- responses from ALL MCP tools made available to the agent: database query
  results, code fragments, paths, object names;
- working paths in call arguments.

The provider stores and uses the received data according to its own policies —
the service has no control over this.

Safe mode, step by step:

1. Work on an isolated copy of the repository, not in the working directory:
   `git worktree` creates a copy only from files under version control — without
   `.env` and other files outside git (secrets committed to the repository will
   be copied). Changes are returned as a diff and applied manually.
2. If the agent is given code search, start a separate code index instance for
   the copy. The shared index knows all projects on the machine and lists them
   with their paths, and that is also sent to the model.
3. Use a copy of the production database, not the production database itself,
   and a read-only account.
4. Restrict `[fs].allowed_roots` to the copy's directory.
5. Narrow the agent's `allowed_tools`: every unnecessary tool is a leakage
   channel.
6. Create a separate provider key with its own billing and quota. Prefer a
   provider with an agreement not to use data for training, or a local model
   through `[providers.direct.<name>]`.

Steps 1 and 2 are performed together by the example `scripts/clean_copy.py` —
for the [code-index](https://github.com/Regsorm/code-index-mcp) (`bsl-indexer`):

```bash
python scripts/clean_copy.py prepare <project> <copy-name> [--language python]
python scripts/clean_copy.py apply   <project> <copy-name>
python scripts/clean_copy.py remove  <project> <copy-name>
```

`prepare` creates a copy in `<AGENT_WORK_DIR>/<copy-name>` (`C:/Temp/agent-work`
by default; on Linux, `agent-work` in the system temporary directory), starts an index at `127.0.0.1:8037` for that copy alone under the
`work` alias, and verifies that index responses contain no unrelated paths (if
shared-index configurations are not found, the script warns that the check is
incomplete). The
indexer is taken from the `CODE_INDEX_EXE` variable; otherwise, `bsl-indexer` is
searched for in PATH. Shared-index configurations for the isolation check are
taken from `CODE_INDEX_MAIN_HOME` (the indexer directory by default). `apply`
transfers changes from the copy to the project without committing, while
`remove` stops the index and deletes the copy.

Only the call result file, service log, and call database remain locally.

## Installation

### 1. Build

```bash
cd <repository dir>
cargo build --release
```

Binary: `target/release/agents-mcp.exe`. On Windows with GNU build tools,
`dlltool` must be in PATH (for example, from w64devkit).

**Linux** (tested on Ubuntu 22.04, x86-64): the same command produces
`target/release/agents-mcp`; from a Windows machine, build with `cargo zigbuild --release
--target x86_64-unknown-linux-gnu`. glibc 2.30 or newer is required. Set paths in
the configuration (`sqlite_path`, `log_dir`, `runs_dir`, `agents_dir`,
`[fs].allowed_roots`) explicitly on Linux: without them, `logs`, `runs`, and
`agents-mcp.db` are placed in the process's current directory, and file tools
are allowed in `/tmp`. The service shuts down gracefully on SIGTERM and is suitable for a
systemd unit.

### 2. Layout

```powershell
$repo = "<repository dir>"
$dst  = "C:\agents-mcp"
New-Item -ItemType Directory -Force -Path "$dst\bin", "$dst\configs", "$dst\agents" | Out-Null
Copy-Item "$repo\target\release\agents-mcp.exe" -Destination "$dst\bin\" -Force
Copy-Item "$repo\configs\agents-mcp.example.toml" -Destination "$dst\configs\agents-mcp.toml" -Force
```

Then enter your own values in `configs\agents-mcp.toml`: paths to Claude Code CLI
and Codex CLI (or remove their sections), and `[fs].allowed_roots`.

### 3. Key and Login

The DeepSeek key is stored in the environment variable named in `api_key_env`:

```powershell
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', '<your-key>', 'User')
```

Instead of user variables, you can place a `.env` file in the service's working
directory: it is read at startup and re-read together with the configuration.

Claude Code CLI and Codex CLI do not require a key — they authenticate through
their own login: the `/login` command in Claude Code and `codex login` in Codex
CLI.

### 4. Start

```powershell
C:\agents-mcp\bin\agents-mcp.exe --config C:\agents-mcp\configs\agents-mcp.toml
```

The service listens on `http://127.0.0.1:8025/mcp`. For a client that does not
support HTTP, use `--transport stdio`. It can be run in the background by any
service manager.

## Health Check and Log

```powershell
# Health check (without an MCP handshake)
curl http://127.0.0.1:8025/health

# Service log event feed (last 20 records)
# Levels in the level column: 4 — error, 3 — warn, 2 — info.
sqlite3 C:\agents-mcp\logs\agents-mcp.logs.db `
  "SELECT datetime(ts/1000,'unixepoch'), level, target, message FROM events ORDER BY id DESC LIMIT 20"

# Agent call history
sqlite3 C:\agents-mcp\data\agents-mcp.sqlite `
  "SELECT id, agent_name, provider, cost_usd, latency_ms, datetime(created_at, 'unixepoch') FROM agent_calls ORDER BY id DESC LIMIT 20"
```

## Client `.mcp.json` Configuration

```json
"agents": {
  "type": "http",
  "url": "http://127.0.0.1:8025/mcp"
}
```

## Configuration

The complete example with explanations is at
[configs/agents-mcp.example.toml](configs/agents-mcp.example.toml). Minimum:

```toml
[server]
host = "127.0.0.1"
port = 8025
allowed_hosts = ["localhost", "127.0.0.1", "::1"]
# instance = "agents-mcp-1"  # "machine name:port" by default; different for services sharing a database

[storage]
sqlite_path = "../data/agents-mcp.sqlite"  # embedded SQLite (default)
log_dir = "../logs"
runs_dir = "../runs"  # result files of background agent_run calls
# task_store_dsn = "postgres://user:pass@db-host:5432/agents"  # set this to use PostgreSQL

[agents]
agents_dir = "../agents"
hot_reload = true

[providers.direct.deepseek]
api_key_env = "DEEPSEEK_API_KEY"
base_url = "https://api.deepseek.com/v1"

[providers.direct.deepseek.prices."deepseek-flash"]
input = 0.3
output = 1.2
cache_read = 0.006

[providers.direct.anthropic-example]
api = "anthropic"
api_key_env = "ANTHROPIC_EXAMPLE_API_KEY"
base_url = "https://api.example.com/anthropic"
prompt_cache = true

[providers.claude_cli]
executable = "C:/Users/<user>/.local/bin/claude.exe"
max_concurrent = 2
default_max_turns = 8

[providers.codex_cli]
executable = "codex"
codex_home = "C:/Users/<user>/.codex"
max_concurrent = 2
```

For direct HTTP providers, `proxy` accepts an address only with the `http` or
`https` scheme, while `proxy_bypass` supplements the mandatory bypass for
loopback addresses and private networks. For `codex-cli`, these fields work
differently: `proxy` is copied without validation into `HTTP_PROXY`/`HTTPS_PROXY`
for the launched process, while `proxy_bypass` is copied into `NO_PROXY` only
together with `proxy`; there is no automatic bypass for local networks. For
`claude-cli`, the `proxy` and `proxy_bypass` fields do not exist; the process
inherits the service's network environment variables.

Prices are set separately for each HTTP provider model in the
`[providers.<…>.prices."<model>"]` section; the model name must exactly match
`model` in the agent configuration. `input` and `output` are required,
`cache_read` and `cache_write` are optional; all values are dollars per million
tokens. If a cache price is not set, `input` is used. The cost from the provider
response (OpenRouter `usage.cost` or Claude CLI) takes precedence. If the
provider did not report a price and the model is not described in the
configuration, `cost_usd` is `null`, without a warning; tokens are still saved.
Price changes are picked up when the main configuration is reloaded.

### Per-call setting overrides (`overrides`)

`invoke_agent` and the start mode of `agent_run` accept an optional flat
`overrides` object. It applies to one call only, is not added to `input`, and is
not inherited by nested calls. The allowed keys are a closed list:

| Key | Value and validation |
|---|---|
| `model.name` | Non-empty string; the provider is unchanged |
| `model.temperature` | Number from 0 to 2 |
| `model.max_tokens` | Integer greater than 0 |
| `execution.max_turns` | Integer greater than 0 |
| `limits.timeout_sec` | Integer greater than 0 |
| `effort` | Non-empty string; replaces an existing provider effort value |
| `cache.enabled` | Boolean |
| `mcp.<server>.url` | URL of an MCP server already declared by the agent |

An unknown key, invalid type, unknown server, or disallowed URL rejects the
call. Precedence is global `force_provider`/`force_model`, then `overrides`, then
the agent's `config.toml`. When `allowed_mcp_urls` in `[agents]` is non-empty,
the URL must exactly match an entry; without that list, only
`http://127.0.0.1:<port>/mcp` URLs are allowed.

```json
{
  "agent": "code-planner",
  "input": {"task": "Prepare a plan", "work_dir": "C:/Project"},
  "overrides": {
    "execution.max_turns": 60,
    "mcp.code-index.url": "http://127.0.0.1:8037/mcp"
  }
}
```

The chain script has a short option for the same URL:

```powershell
python scripts/code_chain.py --work-dir C:/Project --task "Fix the defect" --code-index-url http://127.0.0.1:8037/mcp
```

### Secrets in configuration

HTTP provider keys are still specified by environment variable name in
`api_key_env`. Configuration string values can use `${NAME}` or
`${NAME:-default value}`. Substitution works in `mcp_config` for `url`, `headers`
values, `env` values, `args` elements, and `command`, as well as in `proxy` for
HTTP providers. For example: `"Authorization": "Bearer ${MY_TOKEN}"`. Values
come first from the process environment and then from the first discovered
`.env`; an empty variable uses the specified default. If a required variable is
missing, the agent call is rejected, while an HTTP provider with such a `proxy`
is not registered.

In `[providers.direct.<name>]` sections, the `api` field selects the API
contract: `"openai"` (default) or `"anthropic"`. For Anthropic, `base_url` is
specified without `/v1`: the `/v1/messages` path is added automatically;
`prompt_cache = true` marks the system prompt and the last tool for ephemeral
caching. This API type does not support live stream recording through
`AGENTS_MCP_STREAM_DIR`, evicting old results from history through
`AGENTS_MCP_HISTORY_TRIM_AT`, or returning the model's skill when the stream
loops.

### Environment Variables

At startup, the service searches for the first `.env` file from the current
directory upward. Process environment variables take precedence over variables
of the same name in `.env`. Values from `.env` that are missing from the
process environment are copied into it at startup.

A configuration reload (`config_reload` or saving the configuration file) also
re-reads `.env`, but only into the service's own map — the process environment
is not changed. New, changed, and deleted values take effect immediately for
keys from `api_key_env` and for `${NAME}` substitutions in `mcp_config` and
`proxy`. The service reads the other variables below from the process
environment: for them, `.env` values are taken as of startup, and changes to the
file take effect only after a restart. An error in an individual `.env` entry is
reported with its line number, without its text.

- `AGENTS_MCP_CONFIG` — path to the main configuration if `--config` was not
  passed.
- `AGENTS_MCP_TASK_STORE_DSN` — a non-empty value overrides
  `[storage].task_store_dsn`.
- The name from each direct provider's `api_key_env` — its API key; without a
  non-empty value, the provider is not registered.
- `RUST_LOG` — the `tracing` log filter.
- `AGENTS_MCP_STREAM_DIR` — enables streaming mode for requests to direct
  OpenAI-compatible providers and sets the live stream recording directory: for
  each call, a pair of files `<time_ms>-<model>.stream.bsl` and `.raw.jsonl` is
  created; records older than 7 days are deleted.
- `AGENTS_MCP_HISTORY_TRIM_AT` — the fraction of the context window after which
  the provider evicts old history; `0.6` by default, zero or a negative value
  disables eviction, and an invalid value is replaced with the default.
- For `claude-cli`, the profile path is taken from `config_dir`, and if it is
  absent, from `CLAUDE_CONFIG_DIR`. The long-lived token is sought first in
  `CLAUDE_CODE_OAUTH_TOKEN`, then in the file specified by
  `CLAUDE_OAUTH_TOKEN_FILE`; if neither exists, `.claude/.credentials.json` is
  sought in `USERPROFILE`, then in `HOME`.

`AGENTS_MCP_TRANSCRIPT=1` is recognized by the provider but currently changes
nothing in the regular service: the runtime always passes a channel that enables
turn recording. The variable has an effect only when the provider is invoked
separately without this channel.

### Reload Without Restart

The main configuration is reloaded at runtime, without restarting the service.
Reloading can be initiated in two ways: save the configuration file (when
`hot_reload = true`, the watcher picks it up) or call the `config_reload` MCP
tool.

Applied at runtime: `[storage] runs_dir`; `[agents]` `force_provider` /
`force_model`, `default_timeout_sec`, `agents_dir`, and `hot_reload`; all of `[providers.*]` (a provider
whose settings have not changed remains the same — with its connections and
semaphore); `[skills]` `rag_query_url`; `[fs] allowed_roots`.

Require a restart and are returned in the `restart_required` list: all of
`[server]` (host, port, allowed_hosts, instance) and `[storage]` `log_dir`,
`sqlite_path`, the external database address (`task_store_dsn`), and
`task_store_pool`.

The `.env` file is re-read together with the configuration; what it applies at
runtime is described in [Environment Variables](#environment-variables). Running calls
finish using the previous provider set; new calls use the new one. A
configuration with a parse error is not applied at all — the response reports
this in `errors`.

### Shutdown Without Interrupting Calls

Stopping the process interrupts running calls, so before doing so, call the
`prepare_shutdown` tool: it stops accepting new calls (`invoke_agent`,
`agent_run` in start mode; the board, history, `wait_agent`, `agent_cancel`, and
`health` continue to work as usual), while `/health` starts returning
`status = "draining"`. Child calls of running orchestrators continue to be
accepted. The `{"status": "draining", ...}` response lists running calls —
repeat `prepare_shutdown` (or set `wait_sec` immediately, up to 55 seconds) until
`status = "ready"`, and only then stop the service. If you change your mind,
`prepare_shutdown` with `abort = true` resumes accepting calls.

## Adding a New Agent

An agent is a directory `agents/<name>/`:

- `config.toml` — the agent description (required);
- `prompt.md` — a Tera prompt template with variables from `input` (required);
- `prompt.<variant>.md` — another prompt variant selected by the `variant`
  parameter at call time (optional);
- `schema.json` — the schema for validating a JSON response; read only when this
  file is specified in `response.schema_file`.

Example `agents/summarizer/config.toml`:

```toml
name = "summarizer"
description = "A concise summary of the text in JSON."
version = "0.1.0"
tags = ["example"]
kind = "prompt-template"      # prompt-template | agent-loop | orchestrator

[model]
provider = "deepseek"         # section name [providers.direct.<name>], "claude-cli", "codex-cli", or "mock"
name = "deepseek-flash"
temperature = 0.2
max_tokens = 1024

[response]
format = "json"               # json | text
schema_file = "schema.json"   # optional; without this field, the schema file is not read

[input]
required = ["text"]
optional = ["max_points"]

[limits]
timeout_sec = 120

[cache]
enabled = true
ttl_sec = 3600
```

Example `agents/summarizer/prompt.md`:

```
Summarize the text concisely. Return JSON: {"points": ["...", "..."]}.
No more than {{ max_points | default(value=5) }} points.

Text:
{{ text }}
```

The agents directory is picked up at runtime after approximately 500 ms (when
`hot_reload = true`).

`kind`: `prompt-template` — one request without tools; `agent-loop` — a loop
with tools; `orchestrator` — an agent that calls other agents itself. The
`[execution]` section sets call execution conditions (`allowed_tools`,
`disallowed_tools`, `permission_mode`, `cwd_template`, `mcp_config`,
`max_turns`, `extra_args`): claude-cli passes them as command-line options,
direct providers pass them as a set of tools in the request to the model, and
codex-cli applies only `extra_args` (the service warns about the other fields
when the agent is loaded). The former section name
`[claude_cli]` continues to be read as an alias. The `allowed_roots` key narrows
the `fs_*` file tools for this agent to its own roots: they apply INSTEAD OF the
service's shared `[fs] allowed_roots`, and the call working directory must reside
inside one of them (without the key, the service's shared list applies).

## License

MIT — see [LICENSE](LICENSE).
