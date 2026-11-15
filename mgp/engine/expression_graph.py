import hashlib


def graph_length(root, limit=None):
    sizes = {}
    pending = [(root, False)]
    while pending:
        item, ready = pending.pop()
        key = id(item)
        if key in sizes:
            continue
        if isinstance(item, str):
            sizes[key] = len(item)
            continue
        if ready:
            lengths = [sizes[id(child)] for child in item[1:]]
            if len(lengths) == 1:
                size = len(item[0]) + lengths[0] + 2
            else:
                size = lengths[0] + lengths[1] + len(item[0]) + 4
            sizes[key] = min(size, limit + 1) if limit is not None else size
        else:
            pending.append((item, True))
            pending.extend((child, False) for child in item[1:] if id(child) not in sizes)
    return sizes[id(root)]


def graph_text(root, max_chars=None):
    if max_chars is not None and graph_length(root, max_chars) > max_chars:
        return None
    texts = {}
    pending = [(root, False)]
    while pending:
        item, ready = pending.pop()
        key = id(item)
        if key in texts:
            continue
        if isinstance(item, str):
            texts[key] = item
        elif ready:
            children = [texts[id(child)] for child in item[1:]]
            texts[key] = f'{item[0]}({children[0]})' if len(children) == 1 else f'({children[0]} {item[0]} {children[1]})'
        else:
            pending.append((item, True))
            pending.extend((child, False) for child in item[1:] if id(child) not in texts)
    return texts[id(root)]


def export_graph(root):
    rows, ids, pending = ([], {}, [(root, False)])
    while pending:
        item, ready = pending.pop()
        key = id(item)
        if key in ids:
            continue
        if isinstance(item, str):
            ids[key] = len(rows)
            rows.append(['atom', item])
        elif ready:
            ids[key] = len(rows)
            rows.append(['op', item[0], *[ids[id(child)] for child in item[1:]]])
        else:
            pending.append((item, True))
            pending.extend((child, False) for child in reversed(item[1:]) if id(child) not in ids)
    return {'nodes': rows, 'root': ids[id(root)]}


def graph_digest(root):
    hashes, pending = ({}, [(root, False)])
    while pending:
        item, ready = pending.pop()
        key = id(item)
        if key in hashes:
            continue
        if isinstance(item, str):
            hashes[key] = hashlib.sha256(b'a' + len(item.encode()).to_bytes(8, 'big') + item.encode()).digest()
        elif ready:
            symbol = item[0].encode()
            hashes[key] = hashlib.sha256(b'o' + len(symbol).to_bytes(2, 'big') + symbol + b''.join(hashes[id(child)] for child in item[1:])).digest()
        else:
            pending.append((item, True))
            pending.extend((child, False) for child in item[1:] if id(child) not in hashes)
    return hashes[id(root)].hex()


def archive_expression(node, max_chars=32768, defer_graph=False):
    expr = getattr(node, '_expression', None)
    if getattr(node, 'evaluation', None) is None:
        text = node.expression
        return (text, None)
    if isinstance(expr, str):
        return (expr if len(expr) <= max_chars else None, (node.evaluation if defer_graph else export_graph(node.evaluation)) if len(expr) > max_chars else None)
    text = graph_text(node.evaluation, max_chars)
    return (text, None if text is not None else (node.evaluation if defer_graph else export_graph(node.evaluation)))
