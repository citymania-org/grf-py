import array
import time
from typing import Union, ByteString

import numpy as np

import grf


N = 1000


def nml_lz77_py(data):
    """
    GRF compression algorithm.

    @param data: Uncompressed data.
    @type  data: C{str}, C{bytearray}, C{bytes} or similar.

    @return: Compressed data.
    @rtype:  C{bytearray}
    """
    stream = data
    position = 0
    output = array.array("B")
    literal_bytes = array.array("B")
    stream_len = len(stream)

    while position < stream_len:
        overlap_len = 0
        start_pos = max(0, position - (1 << 11) + 1)
        # Loop through the lookahead buffer.
        for i in range(3, min(stream_len - position + 1, 16)):
            # Set pattern to find the longest match.
            pattern = stream[position : position + i]
            # Find the pattern match in the window.
            # result = stream.find(pattern, start_pos, position + i - 1)
            result = stream.find(pattern, start_pos, position)
            # If match failed, we've found the longest.
            if result < 0:
                break

            p = position - result
            overlap_len = i
            start_pos = result

        if overlap_len > 0:
            if len(literal_bytes) > 0:
                output.append(len(literal_bytes))
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            val = ((-overlap_len) << 3) & 0xFF | (p >> 8)
            output.append(val)
            output.append(p & 0xFF)
            position += overlap_len
        else:
            literal_bytes.append(stream[position])
            if len(literal_bytes) == 0x80:
                output.append(0)
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            position += 1

    if len(literal_bytes) > 0:
        output.append(len(literal_bytes))
        output.extend(literal_bytes)

    return output.tobytes()


def chatgpt3_nml_lz77_py(data):
    """
    GRF compression algorithm.

    @param data: Uncompressed data.
    @type  data: C{str}, C{bytearray}, C{bytes} or similar.

    @return: Compressed data.
    @rtype:  C{bytearray}
    """
    stream = data
    position = 0
    output = bytearray()
    literal_bytes = bytearray()
    stream_len = len(stream)
    max_window_size = 1 << 11

    while position < stream_len:
        overlap_len = 0
        start_pos = max(0, position - max_window_size + 1)
        best_p = 0
        best_len = 0

        for i in range(3, min(stream_len - position + 1, 16)):
            pattern = stream[position : position + i]
            result = stream.find(pattern, start_pos, position)
            if result < 0:
                break

            p = position - result
            if i > best_len:
                best_len = i
                best_p = p
                overlap_len = i
                start_pos = result

        if overlap_len > 0:
            if literal_bytes:
                output.append(len(literal_bytes))
                output.extend(literal_bytes)
                literal_bytes.clear()

            val = ((-overlap_len) << 3) & 0xFF | (best_p >> 8)
            output.extend([val, best_p & 0xFF])
            position += overlap_len
        else:
            literal_bytes.append(stream[position])
            if len(literal_bytes) == 0x80:
                output.extend([0])
                output.extend(literal_bytes)
                literal_bytes.clear()

            position += 1

    if literal_bytes:
        output.append(len(literal_bytes))
        output.extend(literal_bytes)

    return output


def chatgpt4_nml_lz77_py(data: Union[str, ByteString]) -> bytes:
    """
    GRF compression algorithm.

    @param data: Uncompressed data.
    @type  data: str, bytearray, bytes or similar.

    @return: Compressed data.
    @rtype:  bytes
    """
    MAX_WINDOW_SIZE = 1 << 11
    MAX_LOOKAHEAD_SIZE = 16
    MAX_LITERAL_LEN = 0x80

    stream = data
    position = 0
    output = array.array("B")
    literal_bytes = array.array("B")
    stream_len = len(stream)

    while position < stream_len:
        overlap_len = 0
        start_pos = max(0, position - MAX_WINDOW_SIZE + 1)
        max_lookahead = min(stream_len - position + 1, MAX_LOOKAHEAD_SIZE)

        for i in range(3, max_lookahead):
            pattern = stream[position : position + i]
            result = stream.find(pattern, start_pos, position)

            if result < 0:
                break

            p = position - result
            overlap_len = i
            start_pos = result

        if overlap_len > 0:
            if literal_bytes:
                output.append(len(literal_bytes))
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            val = ((-overlap_len) << 3) & 0xFF | (p >> 8)
            output.append(val)
            output.append(p & 0xFF)
            position += overlap_len
        else:
            literal_bytes.append(stream[position])

            if len(literal_bytes) == MAX_LITERAL_LEN:
                output.append(0)
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            position += 1

    if literal_bytes:
        output.append(len(literal_bytes))
        output.extend(literal_bytes)

    return output.tobytes()


def nml_lz77_py_w_hash(data):
    """
    GRF compression algorithm.

    @param data: Uncompressed data.
    @type  data: C{str}, C{bytearray}, C{bytes} or similar.

    @return: Compressed data.
    @rtype:  C{bytearray}
    """
    stream = data
    position = 0
    output = array.array("B")
    literal_bytes = array.array("B")
    stream_len = len(stream)

    a = np.frombuffer(data, dtype=np.uint8)

    P = 251
    hdata = (a.astype(np.uint64) * P + np.roll(a, -1)) * P + np.roll(a, -2)

    hashes = {}
    prev = np.full(len(a), len(a), dtype=np.uint32)
    for i in range(len(hdata) - 3, -1, -1):
        prev[i] = hashes.get(hdata[i], len(a))
        hashes[hdata[i]] = i

    cur_start_pos = - (1 << 11) + 1

    while position < stream_len:
        overlap_len = 0
        start_pos = max(0, cur_start_pos)

        max_len = min(stream_len - position, 15)
        if max_len > 2:
            #start_pos = max(hashes.get(hdata[position]), start_pos)
            start_pos = hashes[hdata[position]]
            # assert start_pos > position - (1 << 11)
            if start_pos <= position - 3:
                # Loop through the lookahead buffer.
                for i in range(3, max_len + 1):
                    # Set pattern to find the longest match.
                    pattern = stream[position : position + i]
                    # Find the pattern match in the window.
                    # result = stream.find(pattern, start_pos, position + i - 1)
                    result = stream.find(pattern, start_pos, position)
                    # If match failed, we've found the longest.
                    if result < 0:
                        break

                    p = position - result
                    overlap_len = i
                    start_pos = result

        if overlap_len > 0:
            if len(literal_bytes) > 0:
                output.append(len(literal_bytes))
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            val = ((-overlap_len) << 3) & 0xFF | (p >> 8)
            output.append(val)
            output.append(p & 0xFF)
        else:
            literal_bytes.append(stream[position])
            if len(literal_bytes) == 0x80:
                output.append(0)
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            overlap_len = 1

        for _ in range(overlap_len):
            if cur_start_pos >= 0:
                hashes[hdata[cur_start_pos]] = prev[cur_start_pos]
            cur_start_pos += 1
            position += 1

    if len(literal_bytes) > 0:
        output.append(len(literal_bytes))
        output.extend(literal_bytes)

    return output.tobytes()


def nml_lz77_py_w_idx(data):
    """
    GRF compression algorithm.

    @param data: Uncompressed data.
    @type  data: C{str}, C{bytearray}, C{bytes} or similar.

    @return: Compressed data.
    @rtype:  C{bytearray}
    """
    stream = data
    position = 0
    output = array.array("B")
    literal_bytes = array.array("B")
    stream_len = len(stream)

    a = np.frombuffer(data, dtype=np.uint8)

    P = 251
    hdata = (a.astype(np.uint64) * P + np.roll(a, -1)) * P + np.roll(a, -2)

    hashes = np.zeros(256, dtype=np.uint8)
    prev = [None] * len(a)
    for i in range(len(hdata) - 3, -1, -1):
        prev[i] = hashes.get(hdata[i])
        hashes[hdata[i]] = i

    cur_start_pos = - (1 << 11) + 1

    while position < stream_len:
        overlap_len = 0
        start_pos = max(0, cur_start_pos)

        max_len = min(stream_len - position, 15)
        if max_len > 2:
            #start_pos = max(hashes.get(hdata[position]), start_pos)
            start_pos = hashes[hdata[position]]
            # assert start_pos > position - (1 << 11)
            if start_pos <= position - 3:
                # Loop through the lookahead buffer.
                for i in range(3, max_len + 1):
                    # Set pattern to find the longest match.
                    pattern = stream[position : position + i]
                    # Find the pattern match in the window.
                    # result = stream.find(pattern, start_pos, position + i - 1)
                    result = stream.find(pattern, start_pos, position)
                    # If match failed, we've found the longest.
                    if result < 0:
                        break

                    p = position - result
                    overlap_len = i
                    start_pos = result

        if overlap_len > 0:
            if len(literal_bytes) > 0:
                output.append(len(literal_bytes))
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            val = ((-overlap_len) << 3) & 0xFF | (p >> 8)
            output.append(val)
            output.append(p & 0xFF)
        else:
            literal_bytes.append(stream[position])
            if len(literal_bytes) == 0x80:
                output.append(0)
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            overlap_len = 1

        for _ in range(overlap_len):
            if cur_start_pos >= 0:
                hashes[hdata[cur_start_pos]] = prev[cur_start_pos]
            cur_start_pos += 1
            position += 1

    if len(literal_bytes) > 0:
        output.append(len(literal_bytes))
        output.extend(literal_bytes)

    return output.tobytes()


def hashlz77(data):
    P = 251
    a = np.frombuffer(data, dtype=np.uint8)
    h = a.astype(np.uint64) * P + np.roll(a, -1)
    hashes = [None, None, None]
    for i in range(2, 15):
        h *= P
        h += np.roll(a, -i)
        #h = h * P + np.roll(a, -i)
        hashes.append(set(h[:-i]))

    stream = data
    position = 0
    output = array.array("B")
    literal_bytes = array.array("B")
    stream_len = len(stream)

    U64MAX = 2**64
    while position < stream_len:
        overlap_len = 0
        result = -1
        if position < stream_len - 2:
            start_pos = max(0, position - (1 << 11) + 1)
            h = (a[position] * P * P + a[position + 1] * P + a[position + 2]) % U64MAX
            for i in range(3, min(stream_len - position + 1, 16)):
                if not h in hashes[i]:
                    break

                overlap_len = i

        if overlap_len > 0:
            pattern = stream[position : position + overlap_len]
            # result = stream.find(pattern, start_pos, position + i - 1)
            result = stream.find(pattern, start_pos, position)

        if result >= 0:
            if len(literal_bytes) > 0:
                output.append(len(literal_bytes))
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            p = position - result
            val = ((-overlap_len) << 3) & 0xFF | (p >> 8)
            output.append(val)
            output.append(p & 0xFF)
            position += overlap_len
        else:
            literal_bytes.append(stream[position])
            if len(literal_bytes) == 0x80:
                output.append(0)
                output.extend(literal_bytes)
                literal_bytes = array.array("B")

            position += 1

    if len(literal_bytes) > 0:
        output.append(len(literal_bytes))
        output.extend(literal_bytes)

    return output.tobytes()


DATA = []
for i in range(1, N + 1):
    DATA.append(open(f'dumps/dump{i}', 'rb').read())

total_size = sum(map(len, DATA))
print(f'Total bytes: {total_size}')

e = grf.SpriteEncoder()
EXPECTED = [e.sprite_compress(d) for d in DATA]


def time_algo(name, func):
    t = time.time()
    print(f'{name}: ', end='', flush=True)
    output = []
    l = 0
    for d in DATA:
        r = func(d)
        output.append(r)
        l += len(r)
    res = time.time() - t
    print(f'{res:.2f} sec, {l} bytes', end='', flush=True)
    if output != EXPECTED:
        print(f' (FAILED)', end='', flush=True)
    print()


time_algo('nml', e.sprite_compress)
time_algo('nml_py', nml_lz77_py)
time_algo('chatgpt3_nml_py', chatgpt3_nml_lz77_py)
time_algo('chatgpt4_nml_py', chatgpt4_nml_lz77_py)
time_algo('nml_py_w_hash', nml_lz77_py_w_hash)
# time_algo('hash', hashlz77)
