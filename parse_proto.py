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
 
 
if __name__ == "__main__":
    import sys
    import json
 
    path = sys.argv[1] if len(sys.argv) > 1 else "x.proto"
 
    with open(path, "r") as f:
        f.readline()  # skip the first line
        content = f.read()
 
    trades = parse_all_trade_data(content)
    print(f"Parsed {len(trades)} trade_data record(s)")
 
    # Print the first record as pretty JSON so you can inspect the structure
    if trades:
        print(json.dumps(trades[0], indent=2))
