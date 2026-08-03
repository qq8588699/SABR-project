import re

TOKEN_RE = re.compile(r'''
    \s*(?:
        (?P<lbrace>\{) |
        (?P<rbrace>\}) |
        (?P<string>"(?:[^"\\]|\\.)*") |
        (?P<ident>[A-Za-z_][A-Za-z0-9_./:\-]*) |
        (?P<colon>:)
    )
''', re.VERBOSE)

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
    kind, value = tokens[i]
    if kind == 'string':
        return value[1:-1], i + 1
    if kind == 'lbrace':
        return parse_block(tokens, i + 1)
    if kind == 'ident':
        try:
            if '.' in value:
                return float(value), i + 1
            return int(value), i + 1
        except ValueError:
            return value, i + 1
    raise ValueError(f"Unexpected token {kind}={value!r} at {i}")

def parse_block(tokens, i):
    result = {}
    while i < len(tokens):
        kind, value = tokens[i]
        if kind == 'rbrace':
            return result, i + 1
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
    return result, i

def parse_all_trade_data(text):
    """Parse a file containing multiple top-level `trade_data { ... }` blocks."""
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
            continue
        i += 1
    return records
