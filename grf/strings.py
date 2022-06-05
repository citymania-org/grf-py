from nml.grfstrings import NewGRFString, default_lang

from .common import TRAIN
from .actions import DefineStrings


# b'test\\UE08FTEST\\0D\\UE098hi: \\UE08Etest'
# b'test\xee\x82\x8fTEST\r\xee\x82\x98hi: \xee\x82\x8etest'

def grf_compile_string(s):
    nstr = NewGRFString(s, default_lang, '')
    value = nstr.parse_string('ascii', default_lang, 1, {}).encode('utf-8')
    res = b''

    i = 0
    while i < len(value):
        if value[i] == 92:  # /
            if value[i + 1] in (92, 34):  # / and "
                res += value[i + 1: i + 2]
                i += 2
            elif value[i + 1] == 85:  # U
                res += chr(int(value[i + 2 : i + 6], 16)).encode("utf8")
                i += 6
            else:
                res += bytes((int(value[i + 1 : i + 3], 16),))
                i += 3
        else:
            res += value[i: i + 1]
            i += 1
    return res


class StringManager:
    def __init__(self):
        self._strings = []

    def add(self, string):
        string_id = len(self._strings)
        self._strings.append(grf_compile_string(string))
        return string_id

    def get_actions(self):
        if not self._strings:
            return []
        return [DefineStrings(
            feature=TRAIN,
            offset=0xd000,
            is_generic_offset=True,
            strings=self._strings,
        )]
