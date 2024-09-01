from nml import grfstrings
import nml.expression

from .common import TRAIN, GLOBAL_VAR
from .actions import DefineStrings, Define


class TTDString:
    CARGO_PLURAL_NOTHING = 0x000E
    CARGO_PLURAL_PASSENGERS = 0x000F
    CARGO_PLURAL_COAL = 0x0010
    CARGO_PLURAL_MAIL = 0x0011
    CARGO_PLURAL_OIL = 0x0012
    CARGO_PLURAL_LIVESTOCK = 0x0013
    CARGO_PLURAL_GOODS = 0x0014
    CARGO_PLURAL_GRAIN = 0x0015
    CARGO_PLURAL_WOOD = 0x0016
    CARGO_PLURAL_IRON_ORE = 0x0017
    CARGO_PLURAL_STEEL = 0x0018
    CARGO_PLURAL_VALUABLES = 0x0019
    CARGO_PLURAL_COPPER_ORE = 0x001A
    CARGO_PLURAL_MAIZE = 0x001B
    CARGO_PLURAL_FRUIT = 0x001C
    CARGO_PLURAL_DIAMONDS = 0x001D
    CARGO_PLURAL_FOOD = 0x001E
    CARGO_PLURAL_PAPER = 0x001F
    CARGO_PLURAL_GOLD = 0x0020
    CARGO_PLURAL_WATER = 0x0021
    CARGO_PLURAL_WHEAT = 0x0022
    CARGO_PLURAL_RUBBER = 0x0023
    CARGO_PLURAL_SUGAR = 0x0024
    CARGO_PLURAL_TOYS = 0x0025
    CARGO_PLURAL_SWEETS = 0x0026
    CARGO_PLURAL_COLA = 0x0027
    CARGO_PLURAL_CANDYFLOSS = 0x0028
    CARGO_PLURAL_BUBBLES = 0x0029
    CARGO_PLURAL_TOFFEE = 0x002A
    CARGO_PLURAL_BATTERIES = 0x002B
    CARGO_PLURAL_PLASTIC = 0x002C
    CARGO_PLURAL_FIZZY_DRINKS = 0x002D
    CARGO_SINGULAR_NOTHING = 0x002E
    CARGO_SINGULAR_PASSENGER = 0x002F
    CARGO_SINGULAR_COAL = 0x0030
    CARGO_SINGULAR_MAIL = 0x0031
    CARGO_SINGULAR_OIL = 0x0032
    CARGO_SINGULAR_LIVESTOCK = 0x0033
    CARGO_SINGULAR_GOODS = 0x0034
    CARGO_SINGULAR_GRAIN = 0x0035
    CARGO_SINGULAR_WOOD = 0x0036
    CARGO_SINGULAR_IRON_ORE = 0x0037
    CARGO_SINGULAR_STEEL = 0x0038
    CARGO_SINGULAR_VALUABLES = 0x0039
    CARGO_SINGULAR_COPPER_ORE = 0x003A
    CARGO_SINGULAR_MAIZE = 0x003B
    CARGO_SINGULAR_FRUIT = 0x003C
    CARGO_SINGULAR_DIAMOND = 0x003D
    CARGO_SINGULAR_FOOD = 0x003E
    CARGO_SINGULAR_PAPER = 0x003F
    CARGO_SINGULAR_GOLD = 0x0040
    CARGO_SINGULAR_WATER = 0x0041
    CARGO_SINGULAR_WHEAT = 0x0042
    CARGO_SINGULAR_RUBBER = 0x0043
    CARGO_SINGULAR_SUGAR = 0x0044
    CARGO_SINGULAR_TOY = 0x0045
    CARGO_SINGULAR_SWEETS = 0x0046
    CARGO_SINGULAR_COLA = 0x0047
    CARGO_SINGULAR_CANDYFLOSS = 0x0048
    CARGO_SINGULAR_BUBBLE = 0x0049
    CARGO_SINGULAR_TOFFEE = 0x004A
    CARGO_SINGULAR_BATTERY = 0x004B
    CARGO_SINGULAR_PLASTIC = 0x004C
    CARGO_SINGULAR_FIZZY_DRINK = 0x004D
    PASSENGERS = 0x004F
    TONS = 0x0050
    BAGS = 0x0051
    LITERS = 0x0052
    ITEMS = 0x0053
    CRATES = 0x0054
    QUANTITY_NOTHING = 0x006E
    QUANTITY_PASSENGERS = 0x006F
    QUANTITY_COAL = 0x0070
    QUANTITY_MAIL = 0x0071
    QUANTITY_OIL = 0x0072
    QUANTITY_LIVESTOCK = 0x0073
    QUANTITY_GOODS = 0x0074
    QUANTITY_GRAIN = 0x0075
    QUANTITY_WOOD = 0x0076
    QUANTITY_IRON_ORE = 0x0077
    QUANTITY_STEEL = 0x0078
    QUANTITY_VALUABLES = 0x0079
    QUANTITY_COPPER_ORE = 0x007A
    QUANTITY_MAIZE = 0x007B
    QUANTITY_FRUIT = 0x007C
    QUANTITY_DIAMONDS = 0x007D
    QUANTITY_FOOD = 0x007E
    QUANTITY_PAPER = 0x007F
    QUANTITY_GOLD = 0x0080
    QUANTITY_WATER = 0x0081
    QUANTITY_WHEAT = 0x0082
    QUANTITY_RUBBER = 0x0083
    QUANTITY_SUGAR = 0x0084
    QUANTITY_TOYS = 0x0085
    QUANTITY_SWEETS = 0x0086
    QUANTITY_COLA = 0x0087
    QUANTITY_CANDYFLOSS = 0x0088
    QUANTITY_BUBBLES = 0x0089
    QUANTITY_TOFFEE = 0x008A
    QUANTITY_BATTERIES = 0x008B
    QUANTITY_PLASTIC = 0x008C
    QUANTITY_FIZZY_DRINKS = 0x008D
    ABBREV_NOTHING = 0x008E
    ABBREV_PASSENGERS = 0x008F
    ABBREV_COAL = 0x0090
    ABBREV_MAIL = 0x0091
    ABBREV_OIL = 0x0092
    ABBREV_LIVESTOCK = 0x0093
    ABBREV_GOODS = 0x0094
    ABBREV_GRAIN = 0x0095
    ABBREV_WOOD = 0x0096
    ABBREV_IRON_ORE = 0x0097
    ABBREV_STEEL = 0x0098
    ABBREV_VALUABLES = 0x0099
    ABBREV_COPPER_ORE = 0x009A
    ABBREV_MAIZE = 0x009B
    ABBREV_FRUIT = 0x009C
    ABBREV_DIAMONDS = 0x009D
    ABBREV_FOOD = 0x009E
    ABBREV_PAPER = 0x009F
    ABBREV_GOLD = 0x00A0
    ABBREV_WATER = 0x00A1
    ABBREV_WHEAT = 0x00A2
    ABBREV_RUBBER = 0x00A3
    ABBREV_SUGAR = 0x00A4
    ABBREV_TOYS = 0x00A5
    ABBREV_SWEETS = 0x00A6
    ABBREV_COLA = 0x00A7
    ABBREV_CANDYFLOSS = 0x00A8
    ABBREV_BUBBLES = 0x00A9
    ABBREV_TOFFEE = 0x00AA
    ABBREV_BATTERIES = 0x00AB
    ABBREV_PLASTIC = 0x00AC
    ABBREV_FIZZY_DRINKS = 0x00AD
    TOWN_BUILDING_NAME_TALL_OFFICE_BLOCK_1 = 0x200F
    TOWN_BUILDING_NAME_OFFICE_BLOCK_1 = 0x2010
    TOWN_BUILDING_NAME_SMALL_BLOCK_OF_FLATS_1 = 0x2011
    TOWN_BUILDING_NAME_CHURCH_1 = 0x2012
    TOWN_BUILDING_NAME_LARGE_OFFICE_BLOCK_1 = 0x2013
    TOWN_BUILDING_NAME_TOWN_HOUSES_1 = 0x2014
    TOWN_BUILDING_NAME_HOTEL_1 = 0x2015
    TOWN_BUILDING_NAME_STATUE_1 = 0x2016
    TOWN_BUILDING_NAME_FOUNTAIN_1 = 0x2017
    TOWN_BUILDING_NAME_PARK_1 = 0x2018
    TOWN_BUILDING_NAME_OFFICE_BLOCK_2 = 0x2019
    TOWN_BUILDING_NAME_SHOPS_AND_OFFICES_1 = 0x201A
    TOWN_BUILDING_NAME_MODERN_OFFICE_BUILDING_1 = 0x201B
    TOWN_BUILDING_NAME_WAREHOUSE_1 = 0x201C
    TOWN_BUILDING_NAME_OFFICE_BLOCK_3 = 0x201D
    TOWN_BUILDING_NAME_STADIUM_1 = 0x201E
    TOWN_BUILDING_NAME_OLD_HOUSES_1 = 0x201F
    TOWN_BUILDING_NAME_COTTAGES_1 = 0x2036
    TOWN_BUILDING_NAME_HOUSES_1 = 0x2037
    TOWN_BUILDING_NAME_FLATS_1 = 0x2038
    TOWN_BUILDING_NAME_TALL_OFFICE_BLOCK_2 = 0x2039
    TOWN_BUILDING_NAME_SHOPS_AND_OFFICES_2 = 0x203A
    TOWN_BUILDING_NAME_SHOPS_AND_OFFICES_3 = 0x203B
    TOWN_BUILDING_NAME_THEATER_1 = 0x203C
    TOWN_BUILDING_NAME_STADIUM_2 = 0x203D
    TOWN_BUILDING_NAME_OFFICES_1 = 0x203E
    TOWN_BUILDING_NAME_HOUSES_2 = 0x203F
    TOWN_BUILDING_NAME_CINEMA_1 = 0x2040
    TOWN_BUILDING_NAME_SHOPPING_MALL_1 = 0x2041
    TOWN_BUILDING_NAME_IGLOO_1 = 0x2059
    TOWN_BUILDING_NAME_TEPEES_1 = 0x205A
    TOWN_BUILDING_NAME_TEAPOT_HOUSE_1 = 0x205B
    TOWN_BUILDING_NAME_PIGGY_BANK_1 = 0x205C
    INDUSTRY_NAME_COAL_MINE = 0x4802
    INDUSTRY_NAME_POWER_STATION = 0x4803
    INDUSTRY_NAME_SAWMILL = 0x4804
    INDUSTRY_NAME_FOREST = 0x4805
    INDUSTRY_NAME_OIL_REFINERY = 0x4806
    INDUSTRY_NAME_OIL_RIG = 0x4807
    INDUSTRY_NAME_FACTORY = 0x4808
    INDUSTRY_NAME_PRINTING_WORKS = 0x4809
    INDUSTRY_NAME_STEEL_MILL = 0x480A
    INDUSTRY_NAME_FARM = 0x480B
    INDUSTRY_NAME_COPPER_ORE_MINE = 0x480C
    INDUSTRY_NAME_OIL_WELLS = 0x480D
    INDUSTRY_NAME_BANK = 0x480E
    INDUSTRY_NAME_FOOD_PROCESSING_PLANT = 0x480F
    INDUSTRY_NAME_PAPER_MILL = 0x4810
    INDUSTRY_NAME_GOLD_MINE = 0x4811
    INDUSTRY_NAME_BANK_TROPIC_ARCTIC = 0x4812
    INDUSTRY_NAME_DIAMOND_MINE = 0x4813
    INDUSTRY_NAME_IRON_ORE_MINE = 0x4814
    INDUSTRY_NAME_FRUIT_PLANTATION = 0x4815
    INDUSTRY_NAME_RUBBER_PLANTATION = 0x4816
    INDUSTRY_NAME_WATER_SUPPLY = 0x4817
    INDUSTRY_NAME_WATER_TOWER = 0x4818
    INDUSTRY_NAME_FACTORY_2 = 0x4819
    INDUSTRY_NAME_FARM_2 = 0x481A
    INDUSTRY_NAME_LUMBER_MILL = 0x481B
    INDUSTRY_NAME_COTTON_CANDY_FOREST = 0x481C
    INDUSTRY_NAME_CANDY_FACTORY = 0x481D
    INDUSTRY_NAME_BATTERY_FARM = 0x481E
    INDUSTRY_NAME_COLA_WELLS = 0x481F
    INDUSTRY_NAME_TOY_SHOP = 0x4820
    INDUSTRY_NAME_TOY_FACTORY = 0x4821
    INDUSTRY_NAME_PLASTIC_FOUNTAINS = 0x4822
    INDUSTRY_NAME_FIZZY_DRINK_FACTORY = 0x4823
    INDUSTRY_NAME_BUBBLE_GENERATOR = 0x4824
    INDUSTRY_NAME_TOFFEE_QUARRY = 0x4825
    INDUSTRY_NAME_SUGAR_MINE = 0x4826
    NEWS_INDUSTRY_CONSTRUCTION = 0x482D
    NEWS_INDUSTRY_PLANTED = 0x482E
    NEWS_INDUSTRY_CLOSURE_GENERAL = 0x4832
    NEWS_INDUSTRY_CLOSURE_SUPPLY_PROBLEMS = 0x4833
    NEWS_INDUSTRY_CLOSURE_LACK_OF_TREES = 0x4834
    NEWS_INDUSTRY_PRODUCTION_INCREASE_GENERAL = 0x4835
    NEWS_INDUSTRY_PRODUCTION_INCREASE_COAL = 0x4836
    NEWS_INDUSTRY_PRODUCTION_INCREASE_OIL = 0x4837
    NEWS_INDUSTRY_PRODUCTION_INCREASE_FARM = 0x4838
    NEWS_INDUSTRY_PRODUCTION_DECREASE_GENERAL = 0x4839
    NEWS_INDUSTRY_PRODUCTION_DECREASE_FARM = 0x483A
    ERROR_CAN_T_CONSTRUCT_THIS_INDUSTRY = 0x4830
    ERROR_FOREST_CAN_ONLY_BE_PLANTED = 0x4831
    ERROR_CAN_ONLY_BE_POSITIONED = 0x483B

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
    def __init__(self, prefix=''):
        self.prefix = ''
        self._globals = {}
        self._persistents = {}
        self._langs = []
        self._default_lang = grfstrings.Language(True)
        self._default_lang.langid = grfstrings.DEFAULT_LANGUAGE
        self._next_string_id = 1
        self._named_strings = {}

    def add_global(self, ref):
        assert isinstance(ref, StringRef), ref
        if ref.hash in self._globals:
            return self._globals[ref.hash][0]

        string_id = len(self._globals)
        self._globals[ref.hash] = (string_id, ref)
        return string_id

    def add_persistent(self, ref):
        assert isinstance(ref, StringRef), ref
        if ref.hash in self._persistents:
            return self._persistents[ref.hash][0]

        string_id = len(self._persistents) + 0xdc00
        self._persistents[ref.hash] = (string_id, ref)
        return string_id

    def import_lang_dir(self, lang_dir, default_lang_file='english.lng'):
        grfstrings.langs = []
        grfstrings.default_lang = grfstrings.Language(True)
        grfstrings.default_lang.langid = grfstrings.DEFAULT_LANGUAGE
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

    def get_persistent_actions(self):
        res = []
        for index, ref in self._persistents.values():
            res.extend(ref.get_actions(TRAIN, offset=index, is_generic_offset=True))
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
        full_name = self.prefix + name
        if full_name not in self._default_lang.strings:
            raise KeyError(name)
        return StringRef(self, nml.expression.Identifier(full_name, None), hash(full_name))

    def add(self, value, name=None):
        if isinstance(value, StringRef):
            assert name is None
            return value
        if name is None:
            name = self._named_strings.get(value)
            if name is None:
                name = f'GRFPY_{self._next_string_id}'
                self._next_string_id += 1
                self._named_strings[value] = name
                self._default_lang.handle_text(name, None, value, None)
        else:
            name = self.prefix + name
            self._default_lang.handle_text(name, None, value, None)
        return StringRef(self, nml.expression.Identifier(name, None), hash(name))

    def __call__(self, value):
        return self.add(value)


class StringRef:
    def __init__(self, manager, nmlexpr, hash):
        self.manager = manager
        self.nmlexpr = nmlexpr
        self.hash = hash

    def __hash__(self):
        return self.hash

    def __eq__(self, value):
        return self.hash == value.hash

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

    def get_pairs(self):
        self.manager.set_nml_globals()
        ns = self.string_nmlexpr
        strings = [
            (lang_id, grf_compile_string(grfstrings.get_translation(ns, lang_id))) for lang_id in grfstrings.get_translations(ns)
        ]
        strings = [(0x7F, grf_compile_string(grfstrings.get_translation(ns)))] + strings
        return strings

    def get_default(self):
        self.manager.set_nml_globals()
        return grf_compile_string(grfstrings.get_translation(self.string_nmlexpr))

    def get_actions(self, feature, offset, is_generic_offset=False):
        strings = self.get_pairs()
        return [
            DefineStrings(
                feature=feature,
                offset=offset,
                is_generic_offset=is_generic_offset,
                lang=lang_id,
                strings=[text],
            )
            for lang_id, text in strings
        ]

    def get_global_id(self):
        return self.manager.add_global(self)

    def get_persistent_id(self):
        return self.manager.add_persistent(self)

    def __str__(self):
        self.manager.set_nml_globals()
        return grfstrings.get_translation(self.string_nmlexpr)

    def __repr__(self):
        return 'StringRef({!r})'.format(str(self))

    def __eq__(self, ref):
        return self.manager is ref.manager and self.nmlexpr == ref.nmlexpr
