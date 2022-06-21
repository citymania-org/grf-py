from nml import grfstrings
import nml.expression

from .common import TRAIN, GLOBAL_VAR
from .actions import DefineStrings, Define

# def grf_compile_string(s):
#     nstr = grfstrings.NewGRFString(s, grfstrings.default_lang, '')
#     value = nstr.parse_string('ascii', grfstrings.default_lang, 1, {}).encode('utf-8')
#     res = b''


def grf_compile_string(value):
    if isinstance(value, bytes):
        return value
    assert isinstance(value, str), type(value)

    res = b''
    if not grfstrings.is_ascii_string(value):
        res += bytes((0xC3, 0x9E))
    i = 0
    while i < len(value):
        if value[i] == "\\":
            if value[i + 1] in ("\\", '"'):
                res += value[i + 1 : i + 2].encode('utf-8')
                i += 2
            elif value[i + 1] == "U":
                res += chr(int(value[i + 2 : i + 6], 16)).encode("utf8")
                i += 6
            else:
                res += bytes((int(value[i + 1 : i + 3], 16),))
                i += 3
        else:
            res += value[i].encode('utf-8')
            i += 1
    return res


class StringManager:
    def __init__(self):
        self._globals = {}
        self._langs = grfstrings.langs
        self._default_lang = grfstrings.default_lang
        self._next_string_id = 1
        self._named_strings = {}

    def add_global(self, ref):
        assert isinstance(ref, StringRef), ref
        if ref.hash in self._globals:
            return self._globals[ref.hash][0]

        string_id = len(self._globals)
        self._globals[ref.hash] = (string_id, ref)
        return string_id

    def import_lang_dir(self, lang_dir, default_lang_file='english.lng'):
        grfstrings.langs = []
        grfstrings.read_lang_files(lang_dir, default_lang_file)
        self._langs = grfstrings.langs
        self._default_lang = grfstrings.default_lang

    @staticmethod
    def _gen_lang_table(names, extra_names):
        res = {}
        for name, idx in names.items():
            el = extra_names.get(name)
            if el:
                val = [name] + el
            else:
                val = name
            res[idx] = val
        return res

    def get_actions(self):
        res = []

        for lang_id, lang in self._langs or ():
            props = {}
            if lang.genders is not None:
                props['lang_genders'] = self._gen_lang_table(lang.genders, lang.gender_map)
            if lang.cases is not None:
                props['lang_cases'] = self._gen_lang_table(lang.cases, lang.case_map)
            if lang.plural is not None:
                props['lang_plural'] = lang.plural
            if props:
                res.append(Define(
                    feature=GLOBAL_VAR,
                    id=lang.langid,
                    props=props,
                ))

        if self._globals:
            # TODO combine strings for same lang
            for index, ref in self._globals.values():
                res.extend(ref.get_actions(TRAIN, offset=0xd000 + index, is_generic_offset=True))

        return res

    def set_nml_globals(self):
        grfstrings.langs = self._langs
        grfstrings.default_lang = self._default_lang

    def __getitem__(self, name):
        full_name = 'STR_' + name
        if full_name not in self._default_lang.strings:
            raise KeyError(name)
        return StringRef(self, nml.expression.Identifier(full_name, None), hash(full_name))

    def add(self, value, name=None):
        if name is None:
            name = self._named_strings.get(value)
            if name is None:
                name = f'GRFPY_{self._next_string_id}'
                self._next_string_id += 1
                self._named_strings[value] = name
                self._default_lang.handle_text(name, None, value, None)
        else:
            name = 'STR_' + name
            self._default_lang.handle_text(name, None, value, None)
        return StringRef(self, nml.expression.Identifier(name, None), hash(name))

    def __call__(self, value):
        return self.add(value)


class StringRef:
    def __init__(self, manager, nmlexpr, hash):
        self.manager = manager
        self.nmlexpr = nmlexpr
        self.hash = hash

    @property
    def string_nmlexpr(self):
        if isinstance(self.nmlexpr, nml.expression.String):
            return self.nmlexpr
        return nml.expression.String([self.nmlexpr], None)

    def eval(self, *args):
        assert isinstance(self.nmlexpr, nml.expression.Identifier)
        params = [self.nmlexpr]
        hashes = [self.hash]
        for s in args:
            if isinstance(s, str):
                s = self.manager(s)
            if not isinstance(s, StringRef):
                raise ValueError(type(s))
            hashes.append(s.hash)
            params.append(s.string_nmlexpr)

        ref_hash = hash(tuple(hashes))
        return StringRef(self.manager, nml.expression.String(params, None), ref_hash)

    def get_actions(self, feature, offset, is_generic_offset=False):
        self.manager.set_nml_globals()
        strings = [
            (lang_id, grfstrings.get_translation(self.nmlexpr, lang_id)) for lang_id in grfstrings.get_translations(self.string_nmlexpr)
        ]
        strings = [(0x7F, grfstrings.get_translation(self.string_nmlexpr))] + strings
        return [
            DefineStrings(
                feature=feature,
                offset=offset,
                is_generic_offset=is_generic_offset,
                lang=lang_id,
                strings=[grf_compile_string(text)],
            )
            for lang_id, text in strings
        ]

    def get_global_id(self):
        return self.manager.add_global(self)

    def __str__(self):
        self.manager.set_nml_globals()
        return grfstrings.get_translation(self.string_nmlexpr)
