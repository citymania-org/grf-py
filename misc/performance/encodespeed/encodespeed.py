import array
import ctypes
import gc
import time
from typing import Union, ByteString

import numpy as np

import grf

import grfrs

N = 1000
ziggrflib = ctypes.CDLL('zig/libziggrf.so')

gc.disable()


# Define the argument types and return type of the process_bytes function
ziggrflib.encode_sprite_v4.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
ziggrflib.encode_sprite_v4.restype = ctypes.POINTER(ctypes.c_ubyte)

# Define the argument types for the free_bytes function
# ziggrflib.free_bytes.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t)
# ziggrflib.free_bytes.restype = None


def ziggrf_encode_sprite_v4(data):
    out_len = ctypes.c_size_t()
    data_array = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    output_ptr = ziggrflib.encode_sprite_v4(data_array, len(data), ctypes.byref(out_len))
    output_len = out_len.value
    output_bytes = bytes((output_ptr[i] for i in range(output_len)))
    print('OUTPUT_PTR', output_ptr, output_bytes)
    # lib.free_bytes(output_ptr, output_len)
    return output_bytes


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
        # for i in range(3, min(stream_len - position + 1, 17)):
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
            # pattern = stream[position : position + overlap_len]
            # start_pos += 1
            # while True:
            #     r = stream.find(pattern, start_pos, position)
            #     if r < 0:
            #         break
            #     p = position - r
            #     start_pos = r + 1

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


def printa(name, l):
    print(f'{name:10}', end=' ')
    for x in l:
        print(f'{x:3}', end=' ')
    print()

icount = 0

def gen_max_overlap_v1(stream):
    global icount
    stream_len = len(stream)
    max_overlap = [0] * stream_len
    overlap_ofs = [0] * stream_len
    for position in range(1, stream_len):
        overlap_len = 0
        start_pos = max(0, position - (1 << 11) + 1)
        for i in range(3, min(stream_len - position + 1, 17)):
        # for i in range(3, min(stream_len - position + 1, 16)):
            pattern = stream[position : position + i]
            result = stream.find(pattern, start_pos, position + i - 1)
            # result = stream.find(pattern, start_pos, position)
            if result < 0:
                break

            overlap_len = i
            start_pos = result
        for i in range(3, overlap_len + 1):
            end = position + i - 1
            if i > max_overlap[end]:
                max_overlap[end] = i
                overlap_ofs[end] = position - start_pos
    return max_overlap, overlap_ofs


def gen_max_overlap_v2(stream):
    global icount
    n = len(stream)
    max_overlap = [0] * n
    overlap_ofs = [0] * n
    buf = [0] * n
    prev = [-1] * 256
    for i in range(n):
        buf[i] = prev[stream[i]]
        prev[stream[i]] = i

    for l in range(2, min(17, n)):
        for pos in range(n - 1, 0, -1):
            buf[pos] = -1
            start_pos = max(0, pos - (1 << 11))
            i = buf[pos - 1]
            while i >= start_pos:
                if stream[i + 1] == stream[pos]:
                    buf[pos] = i + 1
                    if l > 2:
                        max_overlap[pos] = l
                        overlap_ofs[pos] = pos - i - 1
                    break

                i = buf[i]

    return max_overlap, overlap_ofs


def gen_max_overlap_v3(stream):
    global icount
    n = len(stream)
    max_overlap = [0] * n
    overlap_ofs = [0] * n
    buf = [0] * (n + 1)
    prev = [0] * 256
    for i in range(n):
        buf[i + 1] = prev[stream[i]]
        prev[stream[i]] = i + 1

    for l in range(2, min(17, n)):
        for pos in range(n - 1, 0, -1):
            buf[pos + 1] = 0
            start_pos = max(0, pos - (1 << 11))
            i = buf[pos]
            while i > start_pos:
                if stream[i] == stream[pos]:
                    buf[pos + 1] = i + 1
                    if l > 2:
                        max_overlap[pos] = l
                        overlap_ofs[pos] = pos - i
                    break

                i = buf[i]

    return max_overlap, overlap_ofs


def optimal_lz77_py(data):
    stream = data
    position = 0
    output = array.array("B")
    literal_bytes = array.array("B")
    stream_len = len(stream)

    # max_overlap2, _ = gen_max_overlap_v1(stream)
    max_overlap, overlap_ofs = gen_max_overlap_v3(stream)
    # assert max_overlap == max_overlap2

    open_len = [1] * stream_len
    min_open = [2] * stream_len
    min_closed = [2] * stream_len
    prev_open = [0] * stream_len
    prev_closed = [-1] * stream_len

    for i in range(1, stream_len):
        # v1 Same result
        # min_open[i] = min_closed[i - 1] + 2
        # open_len[i] = 1
        # if open_len[i - 1] + 1 < 0x80:
        #     if min_open[i - 1] + 1 < min_open[i]:
        #         min_open[i] = min_open[i - 1] + 1
        #         open_len[i] = open_len[i - 1] + 1
        #         prev_open[i] = 0
        #     min_closed[i] = min_open[i]
        # else:
        #     min_closed[i] = min_open[i]
        #     if min_open[i - 1] + 1 < min_closed[i]:
        #         min_closed[i] = min_open[i - 1] + 1
        #         prev_closed[i] = 1

        # v2
        min_closed[i] = min_closed[i - 1] + 2
        open_len[i] = 1
        prev_closed[i] = -1

        nl = open_len[i - 1] + 1
        m = 1 + nl
        if nl <= i:
            m += min_closed[i - nl]
        if m < min_closed[i]:
            min_closed[i] = m
            prev_closed[i] = -nl
            if nl < 0x80:
                open_len[i] = nl

        # v3 Same result
        # min_closed[i] = min_closed[i - 1] + 2
        # open_len[i] = 1
        # for j in range(1, min(0x80 + 1, i + 2)):
        #     m = 1 + j
        #     if j <= i:
        #         m += min_closed[i - j]
        #     if m < min_closed[i]:
        #         min_closed[i] = m
        #         open_len[i] = j

        for j in range(3, max_overlap[i] + 1):
            mc = min_closed[i - j] + 2
            if mc < min_closed[i]:
                min_closed[i] = mc
                prev_closed[i] = j
        # assert min_closed[i] <= min_open[i]

    i = stream_len - 1
    parts = []
    while i >= 0:
        p = prev_closed[i]
        # for v1 and v3
        # if p <= 1:
        #     l = open_len[i] if p == 0 else open_len[i - 1] + 1
        #     parts.append(data[i - l + 1: i + 1])
        #     parts.append(bytes((l & 0x7f,)))
        #     i -= l
        if p < 0:
            l = -p
            parts.append(data[i - l + 1: i + 1])
            parts.append(bytes((l & 0x7f,)))
            i -= l
        else:
            parts.append(bytes((
                0x80 | ((16 - p) << 3) | (overlap_ofs[i] >> 8),
                overlap_ofs[i] & 0xFF,
            )))
            i -= p
    # print()
    # printa('DATA', list(data))
    # printa('OVERLAP', max_overlap)
    # printa('OPEN_LEN', open_len)
    # printa('MIN_OPEN', min_open)
    # printa('MIN_CLOSED', min_closed)
    # print(parts)
    return b''.join(reversed(parts))

    #     if overlap_len > 0:
    #         if len(literal_bytes) > 0:
    #             output.append(len(literal_bytes))
    #             output.extend(literal_bytes)
    #             literal_bytes = array.array("B")

    #         val = ((-overlap_len) << 3) & 0xFF | (p >> 8)
    #         output.append(val)
    #         output.append(p & 0xFF)
    #         position += overlap_len
    #     else:
    #         literal_bytes.append(stream[position])
    #         if len(literal_bytes) == 0x80:
    #             output.append(0)
    #             output.extend(literal_bytes)
    #             literal_bytes = array.array("B")

    #         position += 1

    # if len(literal_bytes) > 0:
    #     output.append(len(literal_bytes))
    #     output.extend(literal_bytes)

    # return output.tobytes()


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
    # DATA.append(open(f'dumps/dump{i}', 'rb').read()[-2500:])
    DATA.append(open(f'dumps/dump{i}', 'rb').read())

DATA.append(bytes(range(256)))

total_size = sum(map(len, DATA))
print(f'Total bytes: {total_size}')

e = grf.WriteContext()
EXPECTED = [e.sprite_compress(d) for d in DATA]


def time_algo(name, func):
    print(f'{name}: ', end='', flush=True)
    s = slice(0, -1)
    # s = slice(869, 870)
    # s = slice(0, 200)
    TEST_DATA = DATA[s]
    # print(len(TEST_DATA[0]))
    TEST_EXPECTED = EXPECTED[s]
    DATA_RUN = TEST_DATA
    # DATA_RUN = TEST_DATA * 5
    t = time.time()
    for d in DATA_RUN:
        func(d)
    res = time.time() - t
    l = 0
    output = []
    for d in TEST_DATA:
        r = func(d)
        l += len(r)
        output.append(r)
    t = time.time()
    NDECODE = 100
    for _ in range(NDECODE):
        for o in output:
            grfrs.decode_sprite(o)
    decres = (time.time() - t) / NDECODE
    print(f'{res:.3f} sec, {l} bytes', end='', flush=True)
    if output != TEST_EXPECTED:
        try:
            if all(grfrs.decode_sprite(a) == grfrs.decode_sprite(b) for a, b in zip(output, TEST_EXPECTED)):
                print(f' (CORRECT)', end='', flush=True)
            else:
                print(f' (FAILED)', end='', flush=True)
                for i in range(len(TEST_DATA)):
                    a, b = output[i], TEST_EXPECTED[i]
                    if grfrs.decode_sprite(a) != grfrs.decode_sprite(b):
                        print(f'  TEST {i}: {len(TEST_DATA[i])}')
                # print()
                # print('EXPEC', list(grfrs.decode_sprite(EXPECTED[0])))
                # print('FOUND', list(grfrs.decode_sprite((output[0]))))
        except:
            print(f' (BROKEN)', end='', flush=True)
    else:
        print(f' (EXACT)', end='', flush=True)
    print(f' decode {decres:.5f} sec')


time_algo('zig-v4', ziggrf_encode_sprite_v4)
time_algo('grf-py-v4', grfrs.encode_sprite_v4)
time_algo('grf-py-v3', grfrs.encode_sprite_v3)
# time_algo('grf-py-v1', grfrs.encode_sprite_v1)
time_algo('grf-py-v1p', grfrs.encode_sprite_v1p)
# time_algo('grf-py-v2p', grfrs.encode_sprite_v2p)
# time_algo('grf-py-v2', grfrs.encode_sprite_v2)
time_algo('truegrf', grfrs.encode_sprite_truegrf)
time_algo('nml', e.sprite_compress)
time_algo('nml_py', nml_lz77_py)
time_algo('optimal_py', optimal_lz77_py)
# time_algo('chatgpt3_nml_py', chatgpt3_nml_lz77_py)
# time_algo('chatgpt4_nml_py', chatgpt4_nml_lz77_py)
# time_algo('nml_py_w_hash', nml_lz77_py_w_hash)
# time_algo('hash', hashlz77)
grfrs.print_globals()
