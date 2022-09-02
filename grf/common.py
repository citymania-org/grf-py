import datetime
import struct

import spectra


ZOOM_4X, ZOOM_NORMAL, ZOOM_2X, ZOOM_8X, ZOOM_16X, ZOOM_32X = range(6)
# ZOOM_OUT_4X, ZOOM_NORMAL, ZOOM_OUT_2X, ZOOM_OUT_8X, ZOOM_OUT_16X, ZOOM_OUT_32X = range(6)
BPP_8, BPP_24, BPP_32 = 8, 24, 32

PALETTE = (0, 0, 255, 16, 16, 16, 32, 32, 32, 48, 48, 48, 64, 64, 64, 80, 80, 80, 100, 100, 100, 116, 116, 116, 132, 132, 132, 148, 148, 148, 168, 168, 168, 184, 184, 184, 200, 200, 200, 216, 216, 216, 232, 232, 232, 252, 252, 252, 52, 60, 72, 68, 76, 92, 88, 96, 112, 108, 116, 132, 132, 140, 152, 156, 160, 172, 176, 184, 196, 204, 208, 220, 48, 44, 4, 64, 60, 12, 80, 76, 20, 96, 92, 28, 120, 120, 64, 148, 148, 100, 176, 176, 132, 204, 204, 168, 72, 44, 4, 88, 60, 20, 104, 80, 44, 124, 104, 72, 152, 132, 92, 184, 160, 120, 212, 188, 148, 244, 220, 176, 64, 0, 4, 88, 4, 16, 112, 16, 32, 136, 32, 52, 160, 56, 76, 188, 84, 108, 204, 104, 124, 220, 132, 144, 236, 156, 164, 252, 188, 192, 252, 208, 0, 252, 232, 60, 252, 252, 128, 76, 40, 0, 96, 60, 8, 116, 88, 28, 136, 116, 56, 156, 136, 80, 176, 156, 108, 196, 180, 136, 68, 24, 0, 96, 44, 4, 128, 68, 8, 156, 96, 16, 184, 120, 24, 212, 156, 32, 232, 184, 16, 252, 212, 0, 252, 248, 128, 252, 252, 192, 32, 4, 0, 64, 20, 8, 84, 28, 16, 108, 44, 28, 128, 56, 40, 148, 72, 56, 168, 92, 76, 184, 108, 88, 196, 128, 108, 212, 148, 128, 8, 52, 0, 16, 64, 0, 32, 80, 4, 48, 96, 4, 64, 112, 12, 84, 132, 20, 104, 148, 28, 128, 168, 44, 28, 52, 24, 44, 68, 32, 60, 88, 48, 80, 104, 60, 104, 124, 76, 128, 148, 92, 152, 176, 108, 180, 204, 124, 16, 52, 24, 32, 72, 44, 56, 96, 72, 76, 116, 88, 96, 136, 108, 120, 164, 136, 152, 192, 168, 184, 220, 200, 32, 24, 0, 56, 28, 0, 72, 40, 4, 88, 52, 12, 104, 64, 24, 124, 84, 44, 140, 108, 64, 160, 128, 88, 76, 40, 16, 96, 52, 24, 116, 68, 40, 136, 84, 56, 164, 96, 64, 184, 112, 80, 204, 128, 96, 212, 148, 112, 224, 168, 128, 236, 188, 148, 80, 28, 4, 100, 40, 20, 120, 56, 40, 140, 76, 64, 160, 100, 96, 184, 136, 136, 36, 40, 68, 48, 52, 84, 64, 64, 100, 80, 80, 116, 100, 100, 136, 132, 132, 164, 172, 172, 192, 212, 212, 224, 40, 20, 112, 64, 44, 144, 88, 64, 172, 104, 76, 196, 120, 88, 224, 140, 104, 252, 160, 136, 252, 188, 168, 252, 0, 24, 108, 0, 36, 132, 0, 52, 160, 0, 72, 184, 0, 96, 212, 24, 120, 220, 56, 144, 232, 88, 168, 240, 128, 196, 252, 188, 224, 252, 16, 64, 96, 24, 80, 108, 40, 96, 120, 52, 112, 132, 80, 140, 160, 116, 172, 192, 156, 204, 220, 204, 240, 252, 172, 52, 52, 212, 52, 52, 252, 52, 52, 252, 100, 88, 252, 144, 124, 252, 184, 160, 252, 216, 200, 252, 244, 236, 72, 20, 112, 92, 44, 140, 112, 68, 168, 140, 100, 196, 168, 136, 224, 200, 176, 248, 208, 184, 255, 232, 208, 252, 60, 0, 0, 92, 0, 0, 128, 0, 0, 160, 0, 0, 196, 0, 0, 224, 0, 0, 252, 0, 0, 252, 80, 0, 252, 108, 0, 252, 136, 0, 252, 164, 0, 252, 192, 0, 252, 220, 0, 252, 252, 0, 204, 136, 8, 228, 144, 4, 252, 156, 0, 252, 176, 48, 252, 196, 100, 252, 216, 152, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 92, 156, 52, 108, 176, 64, 124, 200, 76, 144, 224, 92, 224, 244, 252, 200, 236, 248, 180, 220, 236, 132, 188, 216, 88, 152, 172, 244, 0, 244, 245, 0, 245, 246, 0, 246, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 76, 24, 8, 108, 44, 24, 144, 72, 52, 176, 108, 84, 210, 146, 126, 252, 60, 0, 252, 84, 0, 252, 104, 0, 252, 124, 0, 252, 148, 0, 252, 172, 0, 252, 196, 0, 64, 0, 0, 255, 0, 0, 48, 48, 0, 64, 64, 0, 80, 80, 0, 255, 255, 0, 32, 68, 112, 36, 72, 116, 40, 76, 120, 44, 80, 124, 48, 84, 128, 72, 100, 144, 100, 132, 168, 216, 244, 252, 96, 128, 164, 68, 96, 140, 255, 255, 255)
to_spectra = lambda r, g, b: spectra.rgb(float(r) / 255., float(g) / 255., float(b) / 255.)
SPECTRA_PALETTE = {i:to_spectra(PALETTE[i * 3], PALETTE[i * 3 + 1], PALETTE[i * 3 + 2]) for i in range(256)}
SAFE_COLOURS = set(range(1, 0xD7))
ALL_COLOURS = set(range(256))
WATER_COLOURS = set(range(0xF5, 0xFF))


WIN_TO_DOS = [
      0,   1,   2,   3,   4,   5,   6,   7,
      8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,
     24,  25,  26,  27,  28,  29,  30,  31,
      6,   7,  34,  35,  36,  37,  38,  39,
      8,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,
     56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,
     72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,
      4,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103,
    104, 105,   5, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135,
      3, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214,   1,
      2, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 227, 228, 229, 230, 231,
    232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244,   9, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 255,
]

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



class FeatureMeta(type):
    def __call__(cls, feature):
        return cls.FEATURES[feature]


class Feature(metaclass=FeatureMeta):
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.constant = name.upper()

    def __repr__(self):
        return self.constant


FeatureMeta.FEATURES = [None] * 0x14
for k, n in ((0x00, 'train'),
             (0x01, 'rv'),
             (0x02, 'ship'),
             (0x03, 'aircraft'),
             (0x04, 'station'),
             (0x05, 'canal'),
             (0x06, 'bridge'),
             (0x07, 'house'),
             (0x08, 'global_var'),
             (0x09, 'industry_tile'),
             (0x0a, 'industry'),
             (0x0b, 'cargo'),
             (0x0c, 'sound_effect'),
             (0x0d, 'airport'),
             (0x0e, 'signal'),
             (0x0f, 'object'),
             (0x10, 'railtype'),
             (0x11, 'airport_tile'),
             (0x12, 'roadtype'),
             (0x13, 'Tramtype'),
            ):
    FeatureMeta.FEATURES[k] = type.__call__(Feature, k, n)

TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, BRIDGE, HOUSE, GLOBAL_VAR, INDUSTRY_TILE, INDUSTRY, CARGO, \
SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, AIRPORT_TILE, ROADTYPE, TRAMTYPE = FeatureMeta.FEATURES

CANAL = RIVER

VEHICLE_FEATURES = (TRAIN, RV, SHIP, AIRCRAFT)


def read_extended_byte(data, offset):
    res = data[offset]
    if res != 0xff:
        return res, offset + 1
    return data[offset + 1] | (data[offset + 2] << 8), offset + 3


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
