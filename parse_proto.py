"""
Generic parser for protobuf text-format data (no schema/.proto file required).
 
Parses files containing one or more top-level `trade_data { ... }` blocks
into plain Python dicts / lists, with validation for balanced braces.
"""
 
import re
 
 
TOKEN_RE = re.compile(r'''
    \s*(?:
        (?P<lbrace>\{) |
        (?P<rbrace>\}) |
        (?P<string>"(?:[^"\\]|\\.)*") |
        (?P<number>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) |
        (?P<ident>[A-Za-z_][A-Za-z0-9_./\-]*) |
        (?P<colon>:)
    )
''', re.VERBOSE)
 
 
def check_braces_balanced(text):
    """
    Sanity check that braces are balanced, ignoring braces inside quoted strings.
    Raises ValueError with a helpful message if unbalanced.
    """
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                raise ValueError(
                    f"Unmatched closing brace '}}' at character index {idx}"
                )
    if depth != 0:
        raise ValueError(
            f"Unbalanced braces: {depth} unclosed '{{' remaining at end of file"
        )
 
 
def tokenize(text):
    pos = 0
    tokens = []
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            pos += 1
            continue
        pos = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        tokens.append((kind, value))
    return tokens
 
 
def parse_value(tokens, i):
    if i >= len(tokens):
        raise ValueError("Unexpected end of input while expecting a value")
 
    kind, value = tokens[i]
    if kind == 'string':
        return value[1:-1], i + 1
    if kind == 'lbrace':
        return parse_block(tokens, i + 1)
    if kind == 'number':
        if '.' in value or 'e' in value or 'E' in value:
            return float(value), i + 1
        return int(value), i + 1
    if kind == 'ident':
        # bare word like BAD, N/A, CAN_PRICE (unquoted enum-style value)
        return value, i + 1
    raise ValueError(f"Unexpected token {kind}={value!r} at token index {i}")
 
 
def parse_block(tokens, i):
    """
    Parse tokens starting right after an opening '{' (or from the start,
    for the implicit top-level block) until the matching '}' is found.
    Returns (dict, next_index).
    """
    result = {}
    while i < len(tokens):
        kind, value = tokens[i]
        if kind == 'rbrace':
            return result, i + 1  # consume the closing brace, block is done
 
        if kind != 'ident':
            raise ValueError(
                f"Expected a field name, got {kind}={value!r} at token index {i}"
            )
 
        field = value
        i += 1
        if i < len(tokens) and tokens[i][0] == 'colon':
            i += 1
 
        val, i = parse_value(tokens, i)
 
        if field in result:
            if isinstance(result[field], list):
                result[field].append(val)
            else:
                result[field] = [result[field], val]
        else:
            result[field] = val
 
    return result, i  # ran out of tokens (only valid for the true top level)
 
 
def parse_all_trade_data(text):
    """
    Parse a file containing one or more top-level `trade_data { ... }` blocks.
    Returns a list of dicts, one per trade_data record.
    """
    check_braces_balanced(text)
 
    tokens = tokenize(text)
    records = []
    i = 0
    while i < len(tokens):
        kind, value = tokens[i]
        if kind == 'ident' and value == 'trade_data':
            i += 1
            if i < len(tokens) and tokens[i][0] == 'colon':
                i += 1
            if i < len(tokens) and tokens[i][0] == 'lbrace':
                block, i = parse_block(tokens, i + 1)
                records.append(block)
            else:
                raise ValueError(
                    f"Expected '{{' after 'trade_data' at token index {i}"
                )
            continue
        i += 1
    return records
 
 
def trades_to_dataframe(trades, multi_level_columns=False):
    """
    Flatten a list of nested trade_data dicts into a pandas DataFrame.
 
    If multi_level_columns is False (default):
        Nested fields become dotted column names, e.g. tol_data.tolerance.status
 
    If multi_level_columns is True:
        Columns become a MultiIndex, e.g. ('tol_data', 'tolerance', 'status')
        Shorter paths are padded with '' so all column tuples have equal length.
    """
    import pandas as pd
 
    df = pd.json_normalize(trades, sep=".")
 
    if multi_level_columns:
        split_cols = [tuple(col.split(".")) for col in df.columns]
        max_depth = max(len(c) for c in split_cols)
        padded_cols = [c + ("",) * (max_depth - len(c)) for c in split_cols]
        df.columns = pd.MultiIndex.from_tuples(padded_cols)
 
    return df
 
 
def parse_file_to_dataframe(path, multi_level_columns=False, skip_first_line=True):
    """
    Parse a single file into a DataFrame. Designed to be called in a
    worker process, so it does all the file I/O + parsing + framing itself
    and returns a plain DataFrame (safe to pass back between processes).
    """
    with open(path, "r") as f:
        if skip_first_line:
            f.readline()
        content = f.read()
 
    trades = parse_all_trade_data(content)
    df = trades_to_dataframe(trades, multi_level_columns=multi_level_columns)
    df.insert(0, "source_file", path)  # track which file each row came from
    return df
 
 
def parse_files_parallel(paths, multi_level_columns=False, skip_first_line=True,
                          max_workers=None):
    """
    Parse many files in parallel using a process pool, then concatenate
    the resulting DataFrames into one.
 
    max_workers=None lets Python pick based on CPU count.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
 
    dfs = []
    errors = []
 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                parse_file_to_dataframe, path, multi_level_columns, skip_first_line
            ): path
            for path in paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                dfs.append(future.result())
            except Exception as exc:
                errors.append((path, exc))
 
    if errors:
        print(f"WARNING: {len(errors)} file(s) failed to parse:")
        for path, exc in errors:
            print(f"  {path}: {exc}")
 
    if not dfs:
        return pd.DataFrame()
 
    combined = pd.concat(dfs, ignore_index=True)
    return combined
 
 
if __name__ == "__main__":
    import sys
    import json
 
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 parse_trade_data.py file1.proto [file2.proto ...] output.csv")
        sys.exit(1)
 
    # Last argument is treated as the output CSV path, everything before it
    # is treated as input files. If only one argument is given, default the
    # output name.
    if len(args) == 1:
        in_paths = [args[0]]
        out_path = "trades.csv"
    else:
        in_paths = args[:-1]
        out_path = args[-1]
 
    if len(in_paths) == 1:
        # Single file: no need for a process pool
        df = parse_file_to_dataframe(in_paths[0], multi_level_columns=False)
    else:
        # Multiple files: parse in parallel, then concatenate
        df = parse_files_parallel(in_paths, multi_level_columns=False)
 
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {out_path}")
 
