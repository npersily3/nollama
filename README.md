# nollama

Lightweight agentic loop infrastructure for Claude and local Ollama projects. Drop it into any project and build tool-calling agents in minutes — no framework required.

> For how the pieces fit together internally, see [nollama/ARCHITECTURE.md](nollama/ARCHITECTURE.md).

## Installation

```bash
pip install nollama             # Claude (Anthropic API) only
pip install "nollama[ollama]"   # + local Ollama support
```

Set your API key for Claude:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quickstart

```python
import anthropic
from nollama import run_loop_claude

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

# 1. Write plain Python functions — they become Claude tools automatically.
def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Fetch current weather for a city.
    Args:
        city:  City name, e.g. 'London'
        units: Temperature units — 'celsius' or 'fahrenheit'
    """
    # your real implementation here
    return {"city": city, "temp": 22, "units": units, "condition": "sunny"}

# 2. Define a system prompt.
SYSTEM = "You are a helpful assistant. Use tools to answer questions accurately."

# 3. Run the loop — it calls tools until it has a final answer.
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
answer = run_loop_claude(client, "claude-sonnet-4-6", SYSTEM, messages, [get_weather])
print(answer)
```

That's it. The framework:
- Converts your functions to Anthropic tool schemas automatically (from type hints + docstrings)
- Runs the tool-call loop until the model returns a final text answer
- Mutates `messages` in place so you keep the full conversation history

---

## Using a local Ollama model

```bash
ollama serve          # start Ollama in a terminal
ollama pull gemma3    # pull a model
```

```python
from nollama import run_loop_local

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
answer = run_loop_local("gemma3", SYSTEM, messages, [get_weather])
print(answer)
```

`run_loop_local` has the same signature as `run_loop_claude` but without the `client` and `max_tokens` parameters.

---

## Writing good tools

Tools are plain functions. Three rules:

**1. Annotate parameters.** The schema builder maps Python types → JSON types. Unannotated parameters default to `string`.

```python
def get_player(name: str, season: int = 2025) -> dict:
```

Supported types: `str`, `int`, `float`, `bool`, `list`, `dict`.

**2. Write a Google-style docstring.** The first paragraph becomes the tool description. The `Args:` block becomes per-parameter descriptions seen by the model.

```python
def get_player(name: str, season: int = 2025) -> dict:
    """
    Fetch season stats for a player.
    Args:
        name:   Full player name, e.g. 'Stephen Curry'
        season: NBA season end-year, defaults to current season
    """
```

**3. Return small dicts.** Tool results go into the context window. Return summaries, IDs, and counts — not raw DataFrames or large lists.

---

## SQLite tools

Pass `db_path` to get four built-in database tools automatically. Large query results are cached server-side — the model only sees a `query_id` and a small preview, keeping the context window lean.

```python
from nollama import run_loop_claude
import anthropic

client = anthropic.Anthropic()
SYSTEM = "You are a data analyst. Use tools to query the database and answer questions."

cache_out = {}
messages  = [{"role": "user", "content": "Which products had the highest revenue last month?"}]
answer    = run_loop_claude(
    client, "claude-sonnet-4-6", SYSTEM, messages, [],
    db_path="my.db", cache_out=cache_out,
)

# Retrieve any cached DataFrame by its query_id for plotting or export.
df = cache_out["q1"]
```

The four tools added automatically:

| Tool | What it does |
|---|---|
| `run_sql_query` | Runs a SELECT, caches the result, returns `query_id` + preview |
| `run_sql_write` | Runs DDL/DML (INSERT, UPDATE, ALTER TABLE) |
| `get_schema` | Returns CREATE TABLE statements |
| `get_available_tables` | Returns table/view names with column lists |

You can mix `db_path` with your own tools — they are merged:

```python
answer = run_loop_claude(client, model, SYSTEM, messages, [my_custom_tool], db_path="my.db")
```

---

## Claude ↔ Ollama toggle pattern

```python
from nollama import run_loop_claude, run_loop_local

USE_CLAUDE   = True
CLAUDE_MODEL = "claude-sonnet-4-6"
LOCAL_MODEL  = "gemma3"

cache_out = {}

def analyze(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if USE_CLAUDE:
        import anthropic
        client = anthropic.Anthropic()
        return run_loop_claude(client, CLAUDE_MODEL, SYSTEM, messages, [], db_path=DB_PATH, cache_out=cache_out)
    else:
        return run_loop_local(LOCAL_MODEL, SYSTEM, messages, [], db_path=DB_PATH, cache_out=cache_out)
```

---

## SQL auto-repair

Give the SQL tools a fix agent and a failed query gets one automatic repair attempt — the broken SQL, the error, and the live schema go to a model that returns corrected SQLite, which is then retried.

Auto-repair is wired in by building the SQL tools yourself with a `fix_fn` and passing them as ordinary tools (instead of using `db_path`):

```python
from nollama.agent_framework import make_sql_tools, make_claude_fix_fn, QueryCache

cache  = QueryCache()
fix_fn = make_claude_fix_fn(client, "claude-sonnet-4-6", db_path="my.db")
sql_tools = make_sql_tools("my.db", cache, fix_fn=fix_fn)

answer = run_loop_claude(client, "claude-sonnet-4-6", SYSTEM, messages, sql_tools)
df = cache.get("q1")   # same QueryCache you passed in
```

Use `make_ollama_fix_fn(db_path, model="gemma3")` for a local repair model. One retry only — a second failure returns both errors.

---

## Dynamic tools (model-authored)

When `allow_create=True` (the default), the model gets a `create_tool` tool and can write its own Python tools mid-session. A tool it creates on one turn is callable on the next, and tools are persisted to `dynamic_tools/dynamic_tools.json` so they survive restarts.

```python
# Tools the model created in past sessions load automatically.
answer = run_loop_claude(client, model, SYSTEM, messages, [my_tool], db_path="my.db")

# Read-only: expose saved tools but don't let the model create new ones.
answer = run_loop_claude(..., allow_create=False)

# Off entirely.
answer = run_loop_claude(..., dynamic_tools_path=None)
```

Override the persistence file with `dynamic_tools_path="path/to/tools.json"`.

> ⚠️ `create_tool` runs model-generated Python via `exec`. It is **not** a sandbox — only enable it with a model and database you trust. Set `allow_create=False` or `dynamic_tools_path=None` for untrusted use. See [ARCHITECTURE.md](nollama/ARCHITECTURE.md#5-dynamic-tool-registry--dynamictoolregistry) for details.

---

## Dependencies

```
anthropic   # Claude API — required for run_loop_claude
pandas      # Required when db_path is used
ollama      # Optional — only for run_loop_local
```
