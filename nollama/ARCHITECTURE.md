# nollama — Architecture

Everything lives in one file: [`agent_framework.py`](agent_framework.py). The package
`__init__` re-exports the two public entry points:

```python
from nollama import run_loop_claude, run_loop_local
```

No classes to subclass, no config objects, no plugin system. You pass plain
Python functions; the framework turns them into tools and runs the model until
it stops asking for tools.

---

## The big picture

```
your functions ──► schema generation ──► agentic loop ──► final text answer
                                              │
            ┌─────────────────────────────────┼─────────────────────────────┐
            │                                  │                             │
      SQL tools (opt.)              dynamic tools (opt.)              query cache (opt.)
      make_sql_tools()             DynamicToolRegistry              QueryCache
```

Both loops (`run_loop_claude`, `run_loop_local`) share the same shape:

1. Build the tool set: your tools + optional SQL tools + optional dynamic tools.
2. Call the model with the conversation and tool schemas.
3. If the model returned a final answer → return its text.
4. Otherwise execute every requested tool, append the results as a new turn,
   and go back to step 2.

The `messages` list is **mutated in place**, so when a loop returns, your caller
holds the complete conversation history.

---

## Components

### 1. Tool schema generation (Claude only)

`function_to_claude_tool(fn)` reflects over a function and produces an Anthropic
tool schema:

- **Types** come from annotations via `_PYTHON_TO_JSON_TYPE`
  (`str→string`, `int→integer`, `float→number`, `bool→boolean`, `list→array`,
  `dict→object`). Unannotated params default to `string`. `Optional[X]` /
  `Union[X, None]` unwraps to `X`.
- **Descriptions** come from the docstring. `_parse_google_docstring` splits it
  into the leading description paragraph and the per-parameter lines under an
  `Args:` block.
- **Required** = every parameter without a default.

Ollama needs none of this — its Python client accepts raw callables directly, so
`run_loop_local` passes the functions through untouched.

### 2. The agentic loop

`run_loop_claude` and `run_loop_local` are deliberately near-duplicates rather
than one abstracted function — the message formats differ enough (Anthropic
content blocks vs. Ollama messages) that sharing code would cost more than it
saves.

Per-iteration tool assembly happens inside an `assemble()` closure that is
re-run every loop turn. This matters because the dynamic-tool registry can grow
**mid-conversation** — a tool the model creates on turn 3 is callable on turn 4.

Tool execution is wrapped in try/except: an exception becomes
`{"error": str(e)}` handed back to the model instead of crashing the loop.
Claude results are JSON-serialized and truncated at `_TOOL_RESULT_CHAR_LIMIT`
(200k chars) to protect the context window.

**Loop differences:**

| | `run_loop_claude` | `run_loop_local` |
|---|---|---|
| Model API | `client.messages.create` | `ollama.chat` |
| Tool schemas | generated dicts | raw callables |
| System prompt | content blocks w/ `cache_control` | injected as `messages[0]` |
| Stop condition | `stop_reason != "tool_use"` | no `tool_calls` in reply |
| Token accounting | yes (printed if `verbose`) | n/a |

### 3. SQLite tools — `make_sql_tools(db_path, cache, fix_fn=None)`

Passing `db_path` to either loop auto-attaches four tools:

| Tool | Purpose |
|---|---|
| `run_sql_query` | SELECT → cached in `QueryCache`, returns `query_id` + preview |
| `run_sql_write` | DDL/DML (INSERT, UPDATE, ALTER TABLE…) |
| `get_schema` | CREATE TABLE statements |
| `get_available_tables` | table/view names with column lists |

Each tool opens and closes its own short-lived `sqlite3.connect` — no shared
connection, no pooling. Fine for the single-threaded agent loop.
*(ponytail: per-call connect is the deliberate simplicity ceiling; add a shared
connection only if profiling shows it matters.)*

### 4. Query cache — `QueryCache`

The key trick for keeping large result sets out of the context window.
`run_sql_query` stores the full DataFrame server-side and returns only a
`query_id`, column list, row count, and a 5-row preview. Results over
`max_rows` (default 500) are truncated. After the loop, pass a `cache_out` dict
to retrieve any DataFrame by id (`cache_out["q1"]`) for plotting or export.

### 5. Dynamic tool registry — `DynamicToolRegistry`

Lets the **model write its own tools at runtime**. When `allow_create=True`
(default), the loop exposes a `create_tool(name, description, python_code,
reason)` tool. The model supplies a function body as a string; the registry
`exec`s it into a restricted namespace (`safe_globals`: `sqlite3`, `pd`, `json`,
and `DB_PATH` when a db is in use) and the resulting callable becomes available
on the next turn.

Tools are persisted to `dynamic_tools/dynamic_tools.json` (override with
`dynamic_tools_path`, or pass `None` to disable). On startup the registry
re-`exec`s every saved tool, so a tool the model invented in one session is
available in all future ones. Set `allow_create=False` to expose saved tools
read-only.

> ⚠️ **Security:** `create_tool` runs arbitrary model-generated Python via
> `exec`. `safe_globals` limits the *injected* namespace but does **not**
> sandbox the interpreter — generated code can still reach builtins and import
> modules. Only enable dynamic tools with a model and database you trust. For
> untrusted use, set `allow_create=False` or `dynamic_tools_path=None`.

### 6. SQL fix agents — `make_claude_fix_fn` / `make_ollama_fix_fn`

Optional auto-repair. Build a `fix_fn(bad_sql, error_message) -> fixed_sql` and
pass it to `make_sql_tools(..., fix_fn=...)`. When a query throws,
`run_sql_query` makes **one** repair attempt: it sends the broken SQL, the error,
and the live schema to a model with SQLite-specific alias rules, then retries the
returned query. One attempt only — a second failure returns both errors.

---

## Data flow: a SQL question end to end

```
user question
   └─► run_loop_claude(db_path="my.db", cache_out={})
         ├─ assemble tools: SQL tools + create_tool
         ├─ model calls get_schema / get_available_tables
         ├─ model calls run_sql_query(sql)
         │     ├─ pd.read_sql → DataFrame
         │     ├─ (on error, if fix_fn) one repair retry
         │     └─ QueryCache.store → {query_id, columns, row_count, preview}
         ├─ model reasons over the preview, maybe queries again
         └─ model returns final text
   └─► cache_out now holds {"q1": df, ...} for plotting/export
```

---

## Design choices worth knowing

- **One file, no framework.** Drop `agent_framework.py` into any project. The
  surface area is two functions.
- **Two loops, not one.** Claude and Ollama speak different message formats;
  duplication is cheaper than the abstraction that would unify them.
- **Mutate `messages` in place** so the caller keeps full history for free.
- **Previews, not payloads.** Large SQL results stay server-side; the model sees
  summaries. This is what makes data-analysis agents affordable.
- **Fail soft.** Tool exceptions become `{"error": ...}` messages the model can
  read and react to, rather than crashing the run.

## Dependencies

| Package | When                             |
|---|----------------------------------|
| `anthropic` | required for `run_loop_claude`   |
| `pandas` | required when `db_path` is used  |
| `ollama` | required ,  for `run_loop_local` |
