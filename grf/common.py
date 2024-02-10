import datetime
import struct


ZOOM_NORMAL, ZOOM_4X, ZOOM_2X, ZOOM_OUT_2X, ZOOM_OUT_4X, ZOOM_OUT_8X = range(6)
# ZOOM_OUT_4X, ZOOM_NORMAL, , ZOOM_OUT_16X, ZOOM_OUT_32X = range(6)
BPP_8, BPP_24, BPP_32 = 8, 24, 32
TEMPERATE, ARCTIC, TROPICAL, TOYLAND = 1, 2, 4, 8
NO_CLIMATE, ALL_CLIMATES = 0, TEMPERATE | ARCTIC | TROPICAL | TOYLAND


af_ZA = 0x1b  # afrikaans,True,Afrikaans,Afrikaans,0,male,
ar_EG = 0x14  # arabic_egypt,True,Arabic (Egypt),Arabic (Egypt),1,,
be_BY = 0x10  # belarusian,True,Belarusian,Беларуская,6,n m f p,n nom m abl gen pre p dat acc f
bg_BG = 0x18  # bulgarian,True,Bulgarian,Български,0,n m f p,n m f p
ca_ES = 0x22  # catalan,True,Catalan,Català,0,Femenin Masculin,
cs_CZ = 0x15  # czech,True,Czech,Čeština,10,n m fp mnp map np f,small ins nom big loc gen voc dat acc
cv_RU = 0x0b  # chuvash,True,Chuvash,Чӑвашла,0,,
cy_GB = 0x0f  # welsh,True,Welsh,Cymraeg,0,,
da_DK = 0x2d  # danish,True,Danish,Dansk,0,,
de_DE = 0x02  # german,True,German,Deutsch,0,n m w p,
el_GR = 0x1e  # greek,True,Greek,Ελληνικά,2,n m f,geniki date subs
en_AU = 0x3d  # english_AU,True,English (AU),English (AU),0,,
en_GB = 0x01  # english,True,English (UK),English (UK),0,,
en_US = 0x00  # english_US,True,English (US),English (US),0,,
eo_EO = 0x05  # esperanto,True,Esperanto,Esperanto,0,,n
es_ES = 0x04  # spanish,True,Spanish,Español (ES),0,m f,
es_MX = 0x55  # spanish_MX,True,Spanish (Mexican),Español (MX),0,m f,
et_EE = 0x34  # estonian,True,Estonian,Eesti keel,0,,in sü g
eu_ES = 0x21  # basque,True,Basque,Euskara,0,,
fa_IR = 0x62  # persian,True,Persian,فارسی,0,,
fi_FI = 0x35  # finnish,True,Finnish,Suomi,0,,
fo_FO = 0x12  # faroese,True,Faroese,Føroyskt,0,n m f,
fr_FR = 0x03  # french,True,French,Français,2,m2 m f,
fy_NL = 0x32  # frisian,True,Frisian,Frysk,0,,
ga_IE = 0x08  # irish,True,Irish,Gaeilge,4,,
gd_GB = 0x13  # gaelic,True,Scottish Gaelic,Gàidhlig,13,m f,dat nom voc gen
gl_ES = 0x31  # galician,True,Galician,Galego,0,n m f,
he_IL = 0x61  # hebrew,True,Hebrew,עברית,0,m f,gen plural singular
hi_IN = 0x17  # hindi,True,Hindi,हिन्दी,0,,
hr_HR = 0x38  # croatian,True,Croatian,Hrvatski,6,female male middle,lok ins nom aku gen dat vok
hu_HU = 0x24  # hungarian,True,Hungarian,Magyar,2,,ba t
id_ID = 0x5a  # indonesian,True,Indonesian,Bahasa Indonesia,1,,
io_IO = 0x06  # ido,True,Ido,Ido,0,,
is_IS = 0x29  # icelandic,True,Icelandic,Íslenska,0,kvenkyn karlkyn hvorugkyn,
it_IT = 0x27  # italian,True,Italian,Italiano,0,m ma f,fp fs ms mp
ja_JP = 0x39  # japanese,True,Japanese,日本語,1,,
ko_KR = 0x3a  # korean,True,Korean,한국어,11,m f,
la_VA = 0x66  # latin,True,Latin,Latina,0,n m fp mp np f,dat abl acc gen
lb_LU = 0x23  # luxembourgish,True,Luxembourgish,Lëtzebuergesch,0,,
lt_LT = 0x2b  # lithuanian,True,Lithuanian,Lietuvių,5,mot vyr,kam ka ko kur kreip kas kuo
lv_LV = 0x2a  # latvian,True,Latvian,Latviešu,3,m f,kas
mk_MK = 0x26  # macedonian,True,Macedonian,Македонски,0,,
mr_IN = 0x11  # marathi,True,Marathi,मराठी,0,,
ms_MY = 0x3c  # malay,True,Malay,Melayu,0,,
mt_MT = 0x09  # maltese,True,Maltese,Malti,12,,
nb_NO = 0x2f  # norwegian_bokmal,True,Norwegian (Bokmal),Norsk (bokmål),0,masculine neuter feminine,small
nl_NL = 0x1f  # dutch,True,Dutch,Nederlands,0,,
nn_NO = 0x0e  # norwegian_nynorsk,True,Norwegian (Nynorsk),Norsk (nynorsk),0,masculine neuter feminine,small
pl_PL = 0x30  # polish,True,Polish,Polski,7,n m f,n m w c d b
pt_BR = 0x37  # brazilian_portuguese,True,Portuguese (Brazilian),Português (BR),2,m f,
pt_PT = 0x36  # portuguese,True,Portuguese,Português,0,n m fp mp f,
ro_RO = 0x28  # romanian,True,Romanian,Română,14,,
ru_RU = 0x07  # russian,True,Russian,Русский,6,n m f p,n nom m abl gen pre p dat acc f
sk_SK = 0x16  # slovak,True,Slovak,Slovenčina,10,m z s,g
sl_SI = 0x2c  # slovenian,True,Slovenian,Slovenščina,8,,t d r
sr_RS = 0x0d  # serbian,True,Serbian,Srpski,6,srednji ženski muški,lok ins nom aku big gen dat vok
sv_SE = 0x2e  # swedish,True,Swedish,Svenska,0,,
ta_IN = 0x0a  # tamil,True,Tamil,தமிழ்,0,,
th_TH = 0x42  # thai,True,Thai,Thai,1,,
tr_TR = 0x3e  # turkish,True,Turkish,Türkçe,1,,tamlanan
uk_UA = 0x33  # ukrainian,True,Ukrainian,Українська,6,m mn s f,z d r
ur_PK = 0x5c  # urdu,True,Urdu,Urdu,0,m f,
vi_VN = 0x54  # vietnamese,True,Vietnamese,Tiếng Việt,1,,
zh_CN = 0x56  # simplified_chinese,True,Chinese (Simplified),简体中文,1,,
zh_TW = 0x0c  # traditional_chinese,True,Chinese (Traditional),繁體中文,1,,
ANY_LANGUAGE = 0x7f


def hex_str(s, n=None):
    add = ''
    if n is not None and len(s) > n:
        s = s[:n - 3]
        add = '...'
    if isinstance(s, (bytes, memoryview)):
        return ':'.join('{:02x}'.format(b) for b in s) + add
    return ':'.join('{:02x}'.format(ord(c)) for c in s) + add


def str_hex(s):
    return bytes(int(x, 16) for x in s.split(':'))


def to_bytes(value):
    if isinstance(value, bytes):
        return value
    return value.encode('utf-8')


def utoi8(value):
    return value | -(value & 0x80)


def utoi32(value):
    return value | -(value & 0x80000000)


def is_leap_year(year):
    return yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)


def date_to_days(d):
    return (d - datetime.date(1, 1, 1)).days + 366

def days_to_date(days):
    if days < 366 or days > 3652424: return days
    # TODO make grf.date
    return datetime.date(1, 1, 1) + datetime.timedelta(days=days-366)


class FeatureMeta(type):
    def __call__(cls, feature):
        return cls.FEATURES[feature]


class Feature(metaclass=FeatureMeta):
    _FROM_NAME = {}

    def __init__(self, id, name, class_name):
        self.id = id
        self.name = name
        self.class_name = class_name
        self.constant = name.upper()

    def __repr__(self):
        return self.constant

    @classmethod
    def from_name(cls, name):
        return cls._FROM_NAME[name]

    @classmethod
    def from_constant(cls, constant):
        cls.from_name(constant.lower())



FeatureMeta.FEATURES = [None] * 0x15
for k, n, cn in (
            (0x00, 'train', 'Train'),
            (0x01, 'rv', 'RV'),
            (0x02, 'ship', 'Ship'),
            (0x03, 'aircraft', 'Aircraft'),
            (0x04, 'station', 'Station'),
            (0x05, 'canal', 'Canal'),
            (0x06, 'bridge', 'Bridge'),
            (0x07, 'house', 'House'),
            (0x08, 'global_var', 'GlobalVar'),
            (0x09, 'industry_tile', 'IndustryTile'),
            (0x0a, 'industry', 'Industry'),
            (0x0b, 'cargo', 'Cargo'),
            (0x0c, 'sound_effect', 'SoundEffect'),
            (0x0d, 'airport', 'Airport'),
            (0x0e, 'signal', 'Signal'),
            (0x0f, 'object', 'Object'),
            (0x10, 'railtype', 'Railtype'),
            (0x11, 'airport_tile', 'AirportTile'),
            (0x12, 'roadtype', 'Roadtype'),
            (0x13, 'tramtype', 'Tramtype'),
            (0x14, 'road_stop', 'RoadStop'),
        ):
    obj = type.__call__(Feature, k, n, cn)
    FeatureMeta.FEATURES[k] = obj
    Feature._FROM_NAME[n] = obj

TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, BRIDGE, HOUSE, GLOBAL_VAR, INDUSTRY_TILE, INDUSTRY, CARGO, \
SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, AIRPORT_TILE, ROADTYPE, TRAMTYPE, ROAD_STOP = FeatureMeta.FEATURES
TOWN = 0xFF  # fake feature for varact2

CANAL = RIVER

VEHICLE_FEATURES = (TRAIN, RV, SHIP, AIRCRAFT)


def read_extended_byte(data, offset):
    res = data[offset]
    if res != 0xff:
        return res, offset + 1
    return data[offset + 1] | (data[offset + 2] << 8), offset + 3


def encode_extended_byte(value):
    if value < 255:
        return struct.pack('<B', value)
    else:
        return struct.pack('<BH', 255, value)


def read_word(data, offset):
    return data[offset] | (data[offset + 1] << 8), offset + 2


def read_dword(data, offset):
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24), offset + 4



class DataReader:
    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset

    def get_byte(self):
        self.offset += 1
        return self.data[self.offset - 1]

    def get_extended_byte(self):
        res, self.offset = read_extended_byte(self.data, self.offset)
        return res

    def get_word(self):
        return self.get_byte() | (self.get_byte() << 8)

    def get_dword(self):
        return self.get_word() | (self.get_word() << 16)

    def get_var(self, n):
        size = 1 << n
        res = struct.unpack_from({0: '<B', 1: '<H', 2: '<I'}[n], self.data, offset=self.offset)[0]
        self.offset += size
        return res

    def hex_str(self, n):
        return hex_str(self.data[self.offset: self.offset + n])


DAYS_FOR_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ACCUM_DAYS_FOR_MONTH = [sum(DAYS_FOR_MONTH[:i]) for i in range(len(DAYS_FOR_MONTH))]


def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_till(year):
    res = 365 * year
    if year > 0:
        y = year - 1
        # add number of leap years till
        res +=  y // 4 - y // 100 + y // 400 + 1
    return res


class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def to_days(self):
        days = ACCUM_DAYS_FOR_MONTH[self.month - 1] + self.day - 1
        if not is_leap_year(self.year) and days >= ACCUM_DAYS_FOR_MONTH[2]:
            days -= 1

        return days_till(self.year) + days


class TTDIndustry:
    COAL_MINE = 0x00
    POWER_PLANT = 0x01
    SAWMILL = 0x02
    FOREST = 0x03
    OIL_REFINERY = 0x04
    OIL_RIG = 0x05
    TEMPERATE_FACTORY = 0x06
    PRINTING_WORKS = 0x07
    STEEL_MILL = 0x08
    TEMPERATE_ARCTIC_FAR = 0x09
    COPPER_ORE_MINE = 0x0A
    OIL_WELLS = 0x0B
    TEMPERATE_BANK = 0x0C
    FOOD_PROCESSING_PLAN = 0x0D
    PAPER_MILL = 0x0E
    GOLD_MINE = 0x0F
    TROPICAL_ARCTIC_BANK = 0x10
    DIAMOND_MINE = 0x11
    IRON_ORE_MINE = 0x12
    FRUIT_PLANTATION = 0x13
    RUBBER_PLANTATION = 0x14
    WATER_WELL = 0x15
    WATER_TOWER = 0x16
    TROPICAL_FACTORY = 0x17
    TROPICAL_FARM = 0x18
    LUMBER_MILL = 0x19
    CANDYFLOSS_FOREST = 0x1A
    SWEETS_FACTORY = 0x1B
    BATTERY_FARM = 0x1C
    COLA_WELLS = 0x1D
    TOY_SHOP = 0x1E
    TOY_FACTORY = 0x1F
    PLASTIC_FOUNTAIN = 0x20
    FIZZY_DRINKS_FACTORY = 0x21
    BUBBLE_GENERATOR = 0x22
    TOFFE_QUARRY = 0x23
    SUGAR_MINE = 0x24

    UNKNOWN = 0xFE
    TOWN = 0xFF


def byte_size_format(num):
    for unit in ("B", "KiB", "MiB"):
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} GiB"


# make sure that numpy array is writable by making copy if it's not
def np_make_writable(nparray):
    if not nparray.flags.writeable:
        return nparray.copy()
    return nparray


class DummyWriteContext:
    class Timer:
        def start(self):
            return self

        def count_loading(self):
            pass

        def count_conversion(self):
            pass

        def count_composing(self):
            pass

        def count_custom(self, category):
            pass

        def print_time_report(self):
            pass

    def __init__(self):
        self.timer = self.Timer()

    def failure(self, obj, message):
        return RuntimeError(message)

    def format_error(self, obj, message):
        return ValueError(message)

    def sanity_warning(self, code, obj, message):
        pass

    def warning(self, code, obj, message):
        pass

    def start_timer(self):
        return self.timer.start()

    def sprite_compress(self, raw_data):
        raise NotImplementedError

    def print_report(self):
        pass
