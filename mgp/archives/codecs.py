import hashlib
import io
import json
import math
import struct
import zlib

import zstandard

MAGIC_METRICS = b'MGPMETR2\x00'
ARCHIVE_FIELDS = ('archive', 'key', 'expression', 'rank', 'matched_id', 'train_rmse', 'clean_rmse', 'score', 'expression_operators', 'expression_leaves', 'expression_depth', 'occurrences', 'used_occurrences', 'tree_count', 'individual_credit', 'inherited_credit', 'root_quality_sum', 'best_root_quality', 'min_depth', 'credit_by_depth', 'domain_guards', 'inherited_self', 'inherited_used', 'expression_graph')


def identity(value):
    if value is None:
        return None
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]


def pack_metrics(metrics):
    metrics = dict(metrics)
    for field in ('selected_expression', 'selected_key', 'episode_selected_expression', 'episode_selected_key'):
        value = metrics.pop(field, None)
        if value is not None:
            metrics[field + '_id'] = identity(value)
    return MAGIC_METRICS + zlib.compress(json.dumps(metrics, ensure_ascii=False, separators=(',', ':'), allow_nan=True, default=str).encode('utf-8'), level=1)


MAGIC = b'MGPIDS1\x00'
NAMES = ('base_fitness', 'bb_snapshot', 'best_bb', 'base_similarity', 'selected_best')


def pack_identity_sets(groups):
    parts = [MAGIC, struct.pack('<B', len(NAMES))]
    for name in NAMES:
        values = groups.get(name, ())
        if any(not isinstance(v, bytes) or len(v) != 16 for v in values):
            raise ValueError('invalid 128-bit identity')
        values = sorted(set(values))
        raw = b''.join(values)
        compressed = zlib.compress(raw, 1) if len(raw) > 256 else raw
        use_compressed = len(compressed) < len(raw)
        payload = compressed if use_compressed else raw
        parts.append(struct.pack('<IBI', len(values), int(use_compressed), len(payload)))
        parts.append(payload)
    return b''.join(parts)


def unpack_identity_sets(blob):
    if not blob.startswith(MAGIC):
        raise ValueError('unknown identity format')
    pos = len(MAGIC)
    count = blob[pos]
    pos += 1
    if count != len(NAMES):
        raise ValueError('unexpected archive count')
    groups = {}
    for name in NAMES:
        if pos + 9 > len(blob):
            raise ValueError('truncated identity header')
        n, compressed, size = struct.unpack_from('<IBI', blob, pos)
        pos += 9
        if compressed not in (0, 1) or size > len(blob) - pos:
            raise ValueError('invalid identity payload')
        payload = blob[pos:pos+size]
        pos += size
        raw = zlib.decompress(payload) if compressed else payload
        if len(raw) != 16*n:
            raise ValueError('identity count mismatch')
        values = {raw[i:i+16] for i in range(0, len(raw), 16)}
        if len(values) != n:
            raise ValueError('duplicate identity values')
        groups[name] = values
    if pos != len(blob):
        raise ValueError('trailing identity bytes')
    return groups


CODEC = 'mgp-json-refs-1'
HEADER = b'MGPJ1\n'


def encode_graph(value):
    sources, records = ([], [])
    containers, strings = ({}, {})

    def reference(item):
        if isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity not in containers:
                containers[identity] = len(sources)
                sources.append(item)
            return {'ref': containers[identity]}
        if isinstance(item, str) and len(item) >= 128:
            if item not in strings:
                strings[item] = len(sources)
                sources.append(item)
            return {'ref': strings[item]}
        if isinstance(item, float) and (not math.isfinite(item)):
            return None
        return item
    root = reference(value)
    index = 0
    while index < len(sources):
        source = sources[index]
        if isinstance(source, dict):
            record = {key: reference(item) for key, item in source.items()}
        elif isinstance(source, (list, tuple)):
            record = [reference(item) for item in source]
        else:
            record = source
        records.append(record)
        index += 1
    return {'codec': CODEC, 'root': root, 'objects': records}


def packed(value):
    encoder = json.JSONEncoder(allow_nan=False, separators=(',', ':'))
    graph = encode_graph(value)
    compressor = zstandard.ZstdCompressor(level=3, write_checksum=True).compressobj()
    output = io.BytesIO()
    output.write(compressor.compress(HEADER))
    prefix = '{"codec":' + encoder.encode(graph['codec']) + ',"root":' + encoder.encode(graph['root']) + ',"objects":['
    chunks, length = ([prefix], len(prefix))
    for index, record in enumerate(graph['objects']):
        chunk = (',' if index else '') + encoder.encode(record)
        chunks.append(chunk)
        length += len(chunk)
        if length >= 65536:
            output.write(compressor.compress(''.join(chunks).encode('utf-8')))
            chunks, length = ([], 0)
    chunks.append(']}')
    output.write(compressor.compress(''.join(chunks).encode('utf-8')))
    output.write(compressor.flush())
    return output.getvalue()


ARCHIVES = ('base_fitness', 'bb_snapshot', 'best_bb')


def archive_members(blob):
    if not isinstance(blob, bytes) or not blob.startswith(b'MGPIDS1\x00'):
        raise ValueError('unsupported or invalid archive membership format')
    data = unpack_identity_sets(blob)
    return {name: data[name] for name in ARCHIVES}
