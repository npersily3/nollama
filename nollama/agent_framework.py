"""
agent_framework.py
==================
Single-file agentic loop infrastructure for Claude and local Ollama projects.

Public API — import from nollama:

    from nollama import run_loop_claude, run_loop_local

Quick start
-----------
    import anthropic
    from nollama import run_loop_claude

    client = anthropic.Anthropic()

    def analyze(question):
        messages = [{"role": "user", "content": question}]
        return run_loop_claude(client, "claude-sonnet-4-6", SYSTEM, messages, [my_tool])

Quick start — with SQLite
--------------------------
    cache_out = {}
    answer = run_loop_claude(
        client, "claude-sonnet-4-6", SYSTEM, messages, [],
        db_path="my.db", cache_out=cache_out,
    )
    df = cache_out["q1"]   # retrieve any cached DataFrame by query_id

Dependencies
------------
    anthropic   — Claude API  (required for run_loop_claude)
    pandas      — required when db_path is used
    ollama      — optional, only for run_loop_local
"""

import inspect
import json
import os
import sqlite3
import typing

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMA GENERATION
# Converts plain Python functions into Anthropic tool-call schema dicts by
# reading type annotations and Google-style docstrings.
# ══════════════════════════════════════════════════════════════════════════════

_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _parse_google_docstring(fn) -> tuple[str, dict[str, str]]:
    """Return (description, {param: description}) from a Google-style docstring."""
    doc = (inspect.getdoc(fn) or "").strip()
    if not doc:
        return "", {}

    lines = doc.splitlines()
    description_lines: list[str] = []
    param_descs: dict[str, str] = {}
    in_args = False
    current_param: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if stripped.endswith(":") and not line.startswith("    ") and in_args:
            in_args = False
            current_param = None
            continue
        if in_args:
            if line.startswith("    ") and ":" in stripped:
                param, _, desc = stripped.partition(":")
                current_param = param.strip()
                param_descs[current_param] = desc.strip()
            elif current_param and stripped:
                param_descs[current_param] = (param_descs[current_param] + " " + stripped).strip()
        else:
            description_lines.append(stripped)

    return " ".join(l for l in description_lines if l), param_descs


def function_to_claude_tool(fn) -> dict:
    """
    Convert a Python function to an Anthropic tool schema dict.

    Reads type annotations for JSON type mapping and a Google-style
    docstring for the tool description and per-parameter descriptions.
    Parameters without defaults are marked required.

    Example
    -------
        def search(query: str, limit: int = 10) -> dict:
            '''
            Search the database.
            Args:
                query: Search term
                limit: Max results
            '''
            ...

        schema = function_to_claude_tool(search)
        # schema["input_schema"]["required"] == ["query"]
    """
    sig = inspect.signature(fn)
    description, param_descs = _parse_google_docstring(fn)
    if not description:
        description = fn.__name__.replace("_", " ").capitalize()

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            json_type = "string"
        else:
            origin = getattr(ann, "__origin__", None)
            if origin is typing.Union:
                non_none = [a for a in ann.__args__ if a is not type(None)]
                inner = non_none[0] if non_none else str
                json_type = _PYTHON_TO_JSON_TYPE.get(inner, "string")
            else:
                json_type = _PYTHON_TO_JSON_TYPE.get(ann, "string")

        prop: dict = {"type": json_type}
        if param_descs.get(param_name):
            prop["description"] = param_descs[param_name]
        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": fn.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _functions_to_claude_tools(fns: list) -> list[dict]:
    return [function_to_claude_tool(fn) for fn in fns]


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC TOOLS — registry construction
# DynamicToolRegistry itself is defined further down; this builds one from a path
# at call time. (Only referenced inside the loops, so forward refs are fine.)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_DYNAMIC_TOOLS_PATH = os.path.join("dynamic_tools", "dynamic_tools.json")


def _make_registry(path: str | None, db_path: str | None):
    """Build a DynamicToolRegistry from a path, loading any saved tools.

    Returns None when path is None (dynamic tools disabled). Created tools get
    sqlite3/pd/json in their namespace, plus DB_PATH when a db is in use.
    """
    if path is None:
        return None
    safe_globals = {"sqlite3": sqlite3, "pd": pd, "json": json}
    if db_path is not None:
        safe_globals["DB_PATH"] = db_path
    registry = DynamicToolRegistry(path, safe_globals)
    registry.load()
    return registry


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE AGENTIC LOOP
# Runs until the model stops calling tools and returns a final text response.
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_RESULT_CHAR_LIMIT = 200_000


def run_loop_claude(
        client,
        model: str,
        system: str | list,
        messages: list[dict],
        tools: list,
        max_tokens: int = 16_000,
        verbose: bool = True,
        db_path: str | None = None,
        cache_out: dict | None = None,
        dynamic_tools_path: str | None = _DEFAULT_DYNAMIC_TOOLS_PATH,
        allow_create: bool = True,
) -> str:
    """
    Run a tool-call loop against the Anthropic API until the model returns
    a final text answer.

    Args:
        client:     An instantiated anthropic.Anthropic client.
        model:      Model string, e.g. 'claude-sonnet-4-6'.
        system:     System prompt string, or a pre-built list of content
                    blocks (e.g. with cache_control already set).
        messages:   The conversation so far. Mutated in-place with assistant
                    turns and tool results so the caller retains full history.
        tools:      List of Python callables. Schemas are built automatically.
        max_tokens: Max tokens per completion call.
        verbose:    Print a token usage summary on completion.
        db_path:    Optional path to a SQLite database. When provided, SQL
                    tools are added automatically.
        cache_out:  Optional dict that is populated with {query_id: DataFrame}
                    entries after the loop finishes. Only used when db_path
                    is set.
        dynamic_tools_path: Path to the JSON file where dynamic tools are
                    persisted. Loaded if it exists, created on first save.
                    Defaults to dynamic_tools/dynamic_tools.json. Pass None
                    to disable dynamic tools entirely.
        allow_create: When True (default), the model gets a create_tool tool
                    and tools it creates mid-session become callable on the
                    next turn. Set False to expose saved dynamic tools
                    read-only.

    Returns:
        The model's final plain-text response.
    """
    if isinstance(system, str):
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_blocks = system

    _cache = None
    if db_path is not None:
        _cache = QueryCache()
        tools = tools + make_sql_tools(db_path, _cache)
    registry = _make_registry(dynamic_tools_path, db_path)
    if registry is not None and allow_create:
        tools = tools + [registry.create_tool_fn()]

    def assemble():
        # registry.callables() can grow mid-loop as the model creates tools
        all_tools = tools + (registry.callables() if registry is not None else [])
        return {fn.__name__: fn for fn in all_tools}, _functions_to_claude_tools(all_tools)

    tool_registry, claude_tools = assemble()

    total_input = total_output = total_cache_read = total_cache_write = 0

    while True:
        if registry is not None:
            tool_registry, claude_tools = assemble()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            tools=claude_tools,
            messages=messages,
        )

        if hasattr(response, "usage"):
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens
            total_cache_read += getattr(response.usage, "cache_read_input_tokens", 0) or 0
            total_cache_write += getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        # ── Final answer ───────────────────────────────────────────────────────
        if response.stop_reason != "tool_use":
            if verbose:
                print(
                    f"\nTokens — input: {total_input:,} | output: {total_output:,}"
                    f" | cache read: {total_cache_read:,} | cache write: {total_cache_write:,}"
                )
            if cache_out is not None and _cache is not None:
                cache_out.update(_cache._cache)
            return " ".join(b.text for b in response.content if b.type == "text")

        # ── Append assistant turn ──────────────────────────────────────────────
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})

        # ── Execute tools and collect results ──────────────────────────────────
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            fn = tool_registry.get(block.name)
            if fn is None:
                result = {"error": f"Unknown tool: {block.name}"}
            else:
                try:
                    result = fn(**block.input)
                except Exception as e:
                    result = {"error": str(e)}

            content = json.dumps(result)
            if len(content) > _TOOL_RESULT_CHAR_LIMIT:
                content = content[:_TOOL_RESULT_CHAR_LIMIT] + "... [TRUNCATED]"
                print(f"[Warning] Tool result from '{block.name}' truncated to {_TOOL_RESULT_CHAR_LIMIT:,} chars")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })

        messages.append({"role": "user", "content": tool_results})


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA AGENTIC LOOP
# Same contract as run_loop but targets a local Ollama instance.
# ══════════════════════════════════════════════════════════════════════════════

def run_loop_local(
        model: str,
        system: str,
        messages: list[dict],
        tools: list,
        verbose: bool = True,
        db_path: str | None = None,
        cache_out: dict | None = None,
        dynamic_tools_path: str | None = _DEFAULT_DYNAMIC_TOOLS_PATH,
        allow_create: bool = True,
) -> str:
    """
    Run a tool-call loop against a local Ollama model until it returns
    a final text answer.

    Requires the `ollama` package and a running Ollama instance (`ollama serve`).
    Ollama's Python client accepts raw callables directly — no schema conversion.

    Args:
        model:     Ollama model name, e.g. 'gemma4', 'llama3', 'mistral'.
        system:    System prompt string.
        messages:  The conversation so far. Mutated in-place.
        tools:     List of Python callables.
        verbose:   Unused; kept for API parity with run_loop_claude.
        db_path:   Optional path to a SQLite database. When provided, SQL
                   tools are added automatically.
        cache_out: Optional dict populated with {query_id: DataFrame} entries
                   after the loop finishes. Only used when db_path is set.
        dynamic_tools_path: Path to the JSON file where dynamic tools are
                   persisted. Loaded if it exists, created on first save.
                   Defaults to dynamic_tools/dynamic_tools.json. Pass None to
                   disable dynamic tools entirely.
        allow_create: When True (default), the model gets a create_tool tool
                   and tools it creates mid-session become callable on the
                   next turn. Set False to expose saved dynamic tools
                   read-only.

    Returns:
        The model's final plain-text response.
    """
    from ollama import chat

    _cache = None
    if db_path is not None:
        _cache = QueryCache()
        tools = tools + make_sql_tools(db_path, _cache)
    registry = _make_registry(dynamic_tools_path, db_path)
    if registry is not None and allow_create:
        tools = tools + [registry.create_tool_fn()]

    def assemble():
        all_tools = tools + (registry.callables() if registry is not None else [])
        return all_tools, {fn.__name__: fn for fn in all_tools}

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system})

    while True:
        active_tools, tool_registry = assemble()
        response = chat(
            model=model,
            messages=messages,
            tools=active_tools,
            options={"temperature": 0.1},
        )

        if not response.message.tool_calls:
            if cache_out is not None and _cache is not None:
                cache_out.update(_cache._cache)
            return response.message.content or ""

        messages.append(response.message)

        for tc in response.message.tool_calls:
            fn = tool_registry.get(tc.function.name)
            fn_args = tc.function.arguments or {}
            if fn is None:
                result = {"error": f"Unknown tool: {tc.function.name}"}
            else:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = {"error": str(e)}

            messages.append({"role": "tool", "content": json.dumps(result)})


# ══════════════════════════════════════════════════════════════════════════════
# SQL FIX AGENTS
# Return a fix_fn(bad_sql, error_message) -> fixed_sql callable.
# Pass the result to make_sql_tools(fix_fn=...) to enable auto-repair.
# ══════════════════════════════════════════════════════════════════════════════

def make_claude_fix_fn(client, model: str, db_path: str):
    """
    Return a fix_fn that uses Claude to repair broken SQLite queries.

    Args:
        client:  An instantiated anthropic.Anthropic client.
        model:   Model string to use for repairs.
        db_path: Path to the SQLite DB (used to fetch the schema).
    """

    def fix_sql_claude(bad_sql: str, error_message: str) -> str:
        conn = sqlite3.connect(db_path)
        schema = "\n".join(
            s[0] for s in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"
            ).fetchall() if s[0]
        )
        conn.close()

        system = [{
            "type": "text",
            "text": (
                "You are a SQLite SQL expert. Fix the broken query below.\n\n"
                "SQLite alias rules:\n"
                "- Aliases CANNOT start with a digit (3P_Index → Three_P_Index)\n"
                "- Aliases with hyphens MUST be wrapped in backticks\n"
                "- ORDER BY alias must exactly match the SELECT alias\n"
                "- Use only letters, digits, and underscores in alias names\n\n"
                f"Database schema:\n{schema}"
            ),
            "cache_control": {"type": "ephemeral"},
        }]
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": (
                f"Fix this SQL.\nSQL: {bad_sql}\nError: {error_message}\n\n"
                "Return ONLY the corrected SQL — no explanation, no markdown fences."
            )}],
        )
        fixed = response.content[0].text.strip()
        if fixed.startswith("```"):
            fixed = fixed.split("```")[1]
            if fixed.startswith("sql"):
                fixed = fixed[3:]
        return fixed.strip()

    return fix_sql_claude


def make_ollama_fix_fn(db_path: str, model: str = "gemma3"):
    """
    Return a fix_fn that uses a local Ollama model to repair broken SQLite queries.

    Requires the `ollama` package and a running Ollama instance.

    Args:
        db_path: Path to the SQLite DB (used to fetch the schema).
        model:   Ollama model name to use for repairs.
    """

    def fix_sql_ollama(bad_sql: str, error_message: str) -> str:
        from ollama import chat

        conn = sqlite3.connect(db_path)
        schema = "\n".join(
            s[0] for s in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"
            ).fetchall() if s[0]
        )
        conn.close()

        response = chat(model=model, messages=[{"role": "user", "content": (
            "You are a SQLite SQL expert. Fix the broken query below.\n\n"
            "SQLite alias rules:\n"
            "- Aliases CANNOT start with a digit (3P_Index → Three_P_Index)\n"
            "- Aliases with hyphens MUST be wrapped in backticks\n"
            "- ORDER BY alias must exactly match the SELECT alias\n\n"
            f"Database schema:\n{schema}\n\n"
            f"SQL: {bad_sql}\nError: {error_message}\n\n"
            "Return ONLY the corrected SQL — no explanation, no markdown fences."
        )}])
        fixed = response.message.content.strip()
        if fixed.startswith("```"):
            fixed = fixed.split("```")[1]
            if fixed.startswith("sql"):
                fixed = fixed[3:]
        return fixed.strip()

    return fix_sql_ollama


# ══════════════════════════════════════════════════════════════════════════════
# QUERY CACHE
# Stores DataFrames server-side. The model receives a query_id and a small
# preview — never the full data — keeping large results out of the context window.
# ══════════════════════════════════════════════════════════════════════════════

class QueryCache:
    """
    Stores DataFrames keyed by auto-incrementing IDs.

    Usage
    -----
        cache = QueryCache(max_rows=500, preview_rows=5)
        qid, meta = cache.store(df)   # meta is safe to return to the model
        df        = cache.get(qid)    # retrieve for visualization
    """

    def __init__(self, max_rows: int = 500, preview_rows: int = 5):
        self.max_rows = max_rows
        self.preview_rows = preview_rows
        self._cache: dict[str, pd.DataFrame] = {}
        self._counter = 0

    def store(self, df: pd.DataFrame) -> tuple[str, dict]:
        """
        Store a DataFrame and return (query_id, metadata).

        Metadata contains columns, capped row count, and a preview —
        suitable for returning directly to the model.
        """
        if len(df) > self.max_rows:
            print(f"[QueryCache] Truncated {len(df)} → {self.max_rows} rows")
            df = df.head(self.max_rows)

        self._counter += 1
        qid = f"q{self._counter}"
        self._cache[qid] = df

        return qid, {
            "query_id": qid,
            "columns": list(df.columns),
            "row_count": len(df),
            "preview": df.head(self.preview_rows).to_dict(orient="records"),
        }

    def get(self, qid: str) -> pd.DataFrame | None:
        return self._cache.get(qid)

    def clear(self):
        self._cache.clear()
        self._counter = 0

    def __contains__(self, qid: str) -> bool:
        return qid in self._cache


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC TOOL REGISTRY
# Lets the model create new Python tools at runtime that persist across
# sessions as a JSON file.
# ══════════════════════════════════════════════════════════════════════════════

class DynamicToolRegistry:
    """
    Manages tools created by the model at runtime.

    Tools are serialised to a JSON file so they survive restarts. On load,
    each tool's source is exec'd into a caller-supplied namespace of safe
    globals (e.g. sqlite3, pandas, your DB_PATH constant).

    Usage
    -----
        registry = DynamicToolRegistry(
            file_path="dynamic_tools.json",
            safe_globals={"sqlite3": sqlite3, "pd": pd, "DB_PATH": DB_PATH},
        )
        registry.load()

        def assemble_tools():
            return base_tools + registry.callables() + [registry.create_tool_fn()]
    """

    def __init__(self, file_path: str, safe_globals: dict):
        self.file_path = file_path
        self.safe_globals = safe_globals
        self._tools: dict[str, dict] = {}  # name → {description, code, reason}
        self._callables: dict[str, callable] = {}

    def load(self):
        """Load and re-register tools saved from prior sessions."""
        if not os.path.exists(self.file_path):
            print("No saved dynamic tools found.")
            return
        with open(self.file_path, "r") as f:
            saved = json.load(f)
        print(f"Loading {len(saved)} saved dynamic tool(s)...")
        for name, entry in saved.items():
            self._register(name, entry["description"], entry["code"], entry["reason"], save=False)

    def _save(self):
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self._tools, f, indent=2)

    def _register(self, name: str, description: str, code: str, reason: str, save: bool = True) -> bool:
        ns = dict(self.safe_globals)
        try:
            exec(code, ns)
            fn = ns.get(name)
            if fn is None:
                print(f"  [Fail] {name}: function not found in provided code")
                return False
            fn.__doc__ = description
            self._callables[name] = fn
            self._tools[name] = {"description": description, "code": code, "reason": reason}
            if save:
                self._save()
            print(f"  [OK] {name}")
            return True
        except Exception as e:
            print(f"  [Fail] {name}: {e}")
            return False

    def callables(self) -> list:
        """Return all registered dynamic tools as a flat list of callables."""
        return list(self._callables.values())

    def create_tool_fn(self):
        """
        Return a create_tool callable to include in assemble_tools().
        The model calls this to register new persistent tools.
        """
        registry = self

        def create_tool(name: str, description: str, python_code: str, reason: str) -> dict:
            """
            Create a new Python tool available for this and future sessions.
            Rules for generated tool code:
            - Only use globals available in the safe namespace.
            - Write results to the database — do NOT return large raw data dicts.
            - Return only a summary (counts, status) to keep the context window small.
            - Skip rows already populated (WHERE col IS NULL) so the tool is safe to re-run.
            Args:
                name: Function name in snake_case (no spaces or hyphens)
                description: What this tool does (used as its docstring)
                python_code: Complete Python function definition as a string
                reason: Why this tool is being created
            Returns:
                Dictionary with status and tool name
            """
            print(f"\n[New Tool] {name}")
            print(f"  Why:  {reason}")
            print(f"  Does: {description}")
            ok = registry._register(name, description, python_code, reason, save=True)
            if ok:
                print(f"  [OK] Registered and saved to {registry.file_path}")
                return {"status": "registered", "tool": name}
            return {"error": f"Failed to register tool '{name}'"}

        return create_tool


# ══════════════════════════════════════════════════════════════════════════════
# SQLITE TOOLS
# Generic database tools. Call make_sql_tools() and drop the result into
# your assemble_tools() list.
# ══════════════════════════════════════════════════════════════════════════════

def make_sql_tools(db_path: str, cache: QueryCache, fix_fn=None) -> list:
    """
    Return [run_sql_query, run_sql_write, get_schema, get_available_tables]
    bound to db_path and cache.

    Args:
        db_path: Path to the SQLite database file.
        cache:   A QueryCache instance.
        fix_fn:  Optional callable(bad_sql, error_message) -> fixed_sql.
                 Build one with make_claude_fix_fn or make_ollama_fix_fn.
                 If provided, run_sql_query attempts one auto-repair on errors.
    """

    def get_schema() -> dict:
        """
        Return the CREATE statements for all tables in the database.
        Returns:
            Dict mapping table name to its CREATE TABLE statement
        """
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        return {name: sql for name, sql in rows if sql}

    def get_available_tables() -> dict:
        """
        List all tables and views in the database with their column names.
        Returns:
            Dict with table/view names mapped to type and column list
        """
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name"
        ).fetchall()
        result = {}
        for name, kind in tables:
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({name})").fetchall()]
            result[name] = {"type": kind, "columns": cols}
        conn.close()
        return result

    def run_sql_query(sql: str) -> dict:
        """
        Run a SELECT query and cache the results server-side. Returns a query_id
        to pass to visualization tools. Always include LIMIT (max 500 rows enforced).
        Args:
            sql: A valid SQLite SELECT query
        Returns:
            query_id, columns, row_count, and a 5-row preview
        """
        print(f"\nSQL:\n{sql}\n")
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(sql, conn)
            conn.close()
        except Exception as e:
            if fix_fn is not None:
                print("SQL failed, spawning fix agent...")
                try:
                    fixed = fix_fn(sql, str(e))
                    print(f"Fixed SQL:\n{fixed}\n")
                    conn = sqlite3.connect(db_path)
                    df = pd.read_sql(fixed, conn)
                    conn.close()
                except Exception as e2:
                    return {"error": f"Original: {e} | After fix attempt: {e2}"}
            else:
                return {"error": str(e)}

        qid, meta = cache.store(df)
        return meta

    def run_sql_write(sql: str) -> dict:
        """
        Execute a write SQL statement (ALTER TABLE, CREATE TABLE, INSERT, UPDATE).
        Use to permanently extend or populate the database.
        Do NOT use for SELECT queries — use run_sql_query instead.
        Args:
            sql: A valid SQLite DDL or DML statement
        Returns:
            Dict with status and rows_affected
        """
        print(f"\nSQL (write):\n{sql}\n")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(sql)
            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()
            print(f"  → {rows_affected} row(s) affected")
            return {"status": "ok", "rows_affected": rows_affected}
        except Exception as e:
            return {"error": str(e)}

    return [run_sql_query, run_sql_write, get_schema, get_available_tables]
