# nollama

Lightweight agentic loop infrastructure for Claude and local Ollama projects. Drop it into any project and build tool-calling agents in minutes — no framework required.

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

## Dependencies

```
anthropic   # Claude API — required for run_loop_claude
pandas      # Required when db_path is used
ollama      # Optional — only for run_loop_local
```
