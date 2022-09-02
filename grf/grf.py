import functools
import heapq
import inspect
import struct
import time
import textwrap
from collections import defaultdict

from PIL import Image, ImageDraw
import nml.spriteencoder
import numpy as np

from .actions import Ref, CB, Range, ReferenceableAction, ReferencingAction, get_ref_id, pformat, PyComment, SpriteRef, \
                     Define, DefineMultiple, Action1, SpriteSet, GenericSpriteLayout, RandomSwitch, \
                     BasicSpriteLayout, AdvancedSpriteLayout, ExtendedSpriteLayout, Switch, \
                     IndustryProductionCallback, Action3, Map, DefineStrings, ReplaceNewSprites, \
                     ModifySprites, If, SetDescription, ReplaceOldSprites, Comment, ComputeParameters, \
                     Label, SoundEffects, ImportSound, Translations, SetProperties
from .parser import Node, Expr, Value, Var, Temp, Perm, Call, parse_code, OP_INIT, SPRITE_FLAGS, GenericVar
from .common import Feature, hex_str, utoi32, FeatureMeta, to_bytes, GLOBAL_VAR
from .common import PALETTE
from .sprites import BaseSprite, GraphicsSprite, SoundSprite, RealSprite, LazyBaseSprite, IntermediateSprite, \
                     PaletteRemap
from .strings import StringManager


TEMPERATE, ARCTIC, TROPICAL, TOYLAND = 1, 2, 4, 8
NO_CLIMATE, ALL_CLIMATES = 0, TEMPERATE | ARCTIC | TROPICAL | TOYLAND


# def map_rgb_image(self, im):
#     assert im.mode == 'RGB', im.mode
#     data = np.array(im)


# class ImageSprite(BaseSprite):
#     def __init__(self, w, h, *, xofs=0, yofs=0, zoom=ZOOM_4X):
#         self.sprite_id = None
#         self.w = w
#         self.h = h
#         self.xofs = xofs
#         self.yofs = yofs
#         self.zoom = zoom

#     def get_data_size(self):
#         return 4

#     def get_data(self):
#         return struct.pack('<I', self.sprite_id)

#     def get_real_data(self, encoder):
#         raise NotImplementedError

#     def draw(self, img):
#         raise NotImplementedError


class SpriteSheet:
    def __init__(self, sprites=None):
        self._sprites = list(sprites) if sprites else []

    def make_image(self, filename, padding=5, columns=10):
        w, h = 0, padding
        lineofs = []
        for i in range(0, len(self._sprites), columns):
            w = max(w, sum(s.w for s in self._sprites[i: i + columns]))
            lineofs.append(h)
            h += padding + max((s.h for s in self._sprites[i: i + columns]), default=0)

        w += (columns + 1) * padding
        im = Image.new('L', (w, h), color=0xff)
        im.putpalette(PALETTE)

        x = 0
        for i, s in enumerate(self._sprites):
            y = lineofs[i // columns]
            if i % columns == 0:
                x = padding
            s.x = x
            s.y = y
            s.file = filename
            s.draw(im)
            x += s.w + padding

        im.save(filename)


class DummySprite(BaseSprite):
    def get_data(self):
        return b'\x00'

    def get_data_size(self):
        return 1


class SpriteGenerator:
    def get_sprites(self, grf):
        return NotImplementedError


class PythonGenerationContext:
    def __init__(self):
        self.resources = {}


class Timer:
    def __init__(self):
        self.t = 0

    def start(self, s):
        self.t = time.time()
        print(f'{s}: ... ', end='', flush=True)

    def log(self, s):
        t = time.time()
        print(f'{t - self.t:.02f}')
        print(f'{s}: ... ', end='', flush=True)
        self.t = t

    def stop(self):
        t = time.time()
        print(f'{t - self.t:.02f}')


class SpriteEncoder:
    def __init__(self):
        self._nml = nml.spriteencoder.SpriteEncoder(True, False, None)
        self.loading_time = 0
        self.conversion_time = 0
        self.composing_time = 0
        self.compression_time = 0

    def count_loading(self, t):
        self.loading_time += t

    def count_conversion(self, t):
        self.conversion_time += t

    def count_composing(self, t):
        self.composing_time += t

    def sprite_compress(self, raw_data):
        t0 = time.time()
        res = self._nml.sprite_compress(raw_data)
        self.compression_time += time.time() - t0
        return res

    def print_time_report(self):
        print(f'   Graphics loading time: {self.loading_time:.02f}')
        print(f'   Graphics conversion time: {self.conversion_time:.02f}')
        print(f'   Graphics composing time: {self.composing_time:.02f}')
        print(f'   Graphics compression time: {self.compression_time:.02f}')


class BaseNewGRF:
    def __init__(self):
        self.generators = []
        self._next_sound_id = 73
        self._sounds = {}
        self.strings = StringManager()
        self._sprite_encoder = SpriteEncoder()

    def _add_sound(self, s):
        assert isinstance(s, SoundSprite)

        if s.id is not None:
            return

        h = s.get_hash()
        if h in self._sounds:
            s.id = self._sounds[h].id
        else:
            if self._next_sound_id > 0xff:
                raise RuntimeError('Too many sound effects (max 183)')
            s.id = self._next_sound_id
            self._next_sound_id += 1
            self._sounds[h] = s

    def _add(self, l, *sprites):
        if not sprites:
            return

        # Unfold real sprite tuple if it was passed as a single arg
        if isinstance(sprites[0], (tuple, list)):
            assert len(sprites) == 1
            sprites = sprites[0]
            assert len(sprites) >= 1
            assert isinstance(sprites[0], GraphicsSprite)

        if isinstance(sprites[0], RealSprite):
            if isinstance(sprites[0], SoundSprite):
                for s in sprites:
                    self._add_sound(s)
                return
            elif isinstance(sprites[0], PaletteRemap):
                l.append((sprites[0],))
                return
            assert(all(isinstance(s, GraphicsSprite) for s in sprites)), sprites
            assert(len(set((s.zoom, s.bpp) for s in sprites)) == len(sprites)), sprites

            l.append(tuple(sprites))
        else:
            l.extend(sprites)

    def add(self, *sprites):
        self._add(self.generators, *sprites)

    def _write_pseudo_sprite(self, f, data, grf_type=0xff):
        f.write(struct.pack('<IB', len(data), grf_type))
        f.write(data)

    def generate_python(self, grf_filename):
        context = PythonGenerationContext()
        res = 'import grf\n'
        make_grf_import = lambda l: 'from grf import ' + ', '.join(l) + '\n'
        res += make_grf_import(f.constant for f in FeatureMeta.FEATURES)
        res += 'from grf import Ref, CB, ImageFile, FileSprite, RAWSound, PaletteRemap\n\n'
        res += 'g = grf.BaseNewGRF()\n\n'
        used_classes = set(x.__class__.__name__ for x in self.generators if not isinstance(x, (tuple, IntermediateSprite)))
        # used_classes.update(('ImageFile', 'ImageSprite', 'RAWSound'))  # TODO check usage of IntermediateSprite
        if used_classes:
            res += '# Bind all the used classes to current grf so they can be used declaratively\n'
            res += ', '.join(used_classes) + ' = ' +  ', '.join(f'g.bind(grf.{c})' for c in used_classes)
            res += '\n'
        res += '\n'

        for s in self.generators:
            if isinstance(s, tuple):
                s = s[0]
            code = textwrap.dedent(s.py(context)).strip()
            res += code + '\n'
            if '\n' in code: res += '\n'

        res += f'\ng.write({grf_filename!r})'
        return res

    def generate_sprites(self):
        res = []

        def run(g):
            for s in g.get_sprites(self):
                if isinstance(s, SpriteGenerator):
                    run(s)
                else:
                    self._add(res, s)

        for g in self.generators:
            if isinstance(g, SpriteGenerator):
                run(g)
            else:
                res.append(g)
        return res

    def resolve_refs(self, sprites):
        refids = {}
        actions = {}
        ordered_refs = defaultdict(list)
        unordered_refs = defaultdict(list)
        ref_count = defaultdict(int)

        def resolve(s, i):
            # Action can reference other actions, resolve all the references
            if isinstance(s, ReferencingAction):
                for r in s.get_refs():
                    if isinstance(r, Ref):
                        if r.is_callback:
                            continue
                        r = refids.get(r.ref_id)
                        if r is None:
                            raise RuntimeError(f'Unresolved direct reference {r} in action {s}')
                        ref_count[id(r)] += 1
                        continue

                    if not isinstance(r, ReferenceableAction):
                        if isinstance(r, SoundSprite):
                            self._add_sound(r)
                        continue

                    if r is s:
                        raise RuntimeError(f'Action {s} references itself')

                    assert isinstance(r, ReferenceableAction)

                    if id(r) not in actions:
                        actions[id(r)] = (r, None)
                        unordered_refs[id(s)].append(id(r))
                    else:
                        ordered_refs[id(s)].append(id(r))
                    ref_count[id(r)] += 1
                    resolve(r, None)

            prev_i = actions.get(id(s), (None, None))[1]
            if prev_i is not None and i is not None and prev_i != i:
                raise RuntimeError(f'Action {s} was added more than once (positions {prev_i} and {i})')
            actions[id(s)] = (s, i)

        for i, s in enumerate(sprites):
            if not isinstance(s, (ReferencingAction, ReferenceableAction)):
                continue

            resolve(s, i)

            # Action can be referenced by other actions, index it
            # Must be done after processing references to allow reusing the same ref_id
            if isinstance(s, ReferenceableAction):
                if s.ref_id is not None:
                    refids[s.ref_id] = s


        ids = list(range(-255, 0))
        reserved_ids = set()
        linked_refs = {}
        visited = set()
        min_ids = len(ids)

        def dfs(aid, linked, feature):
            nonlocal min_ids
            a, ai = actions[aid]
            if a.feature is not None and a.feature != feature:
                raise RuntimeError(f'Mixed features {a.feature} and {feature} in action {a}')
            if aid in visited:
                return
            visited.add(aid)
            if a.feature is None:
                a.feature = feature
            if isinstance(a, ReferenceableAction):
                if a.ref_id is None:
                    try:
                        while a.ref_id is None or a.ref_id in reserved_ids:
                            a.ref_id = -heapq.heappop(ids)
                        min_ids = min(min_ids, len(ids))
                    except IndexError:
                        raise RuntimeError(f'Ran out of ids while trying to reference action {a}')
                else:
                    reserved_ids.add(a.ref_id)

            if ai is not None:
                # Node is already ordered, prepend unordered nodes to it
                linked = linked_refs[aid] = []

            for x in ordered_refs[aid]:
                dfs(x, linked, feature)

            for x in unordered_refs[aid]:
                dfs(x, linked, feature)

            # Apppend node to the ordered parent node (or itself)
            linked.append(a)

            if isinstance(a, ReferencingAction):
                for r in a.get_refs():
                    rid = id(r)
                    # TODO release Ref too?
                    if isinstance(r, ReferenceableAction):
                        if ref_count[rid] == 0:
                            raise RuntimeError('Reference counting error')
                        ref_count[rid] -= 1
                        # If this action is longer referenced return id to the pool
                        if ref_count[rid] == 0:
                            if -r.ref_id not in ids:
                                heapq.heappush(ids, -r.ref_id)
                            reserved_ids.discard(r.ref_id)

        roots = [(r, v[0].feature) for r, v in actions.items() if ref_count[r] <= 0]
        # print('SPRITES', sprites)
        # print('ACTIONS', actions)
        # print('OREF', ordered_refs)
        # print('UREF', unordered_refs)
        # print('roots', roots)
        for r, f in roots:
            assert f is not None
            dfs(r, None, f)

        # print('LREF', linked_refs)

        # print('IDS used', 255 - min_ids)

        # Construct the full list of actions with resolved ids and order
        res = []
        for s in sprites:
            if isinstance(s, ReferencingAction):
                res.extend(linked_refs.get(id(s), []))
            else:
                res.append(s)
        return res


    def write(self, filename):
        t = Timer()
        t.start(f'Evaluating sprite generators')
        sprites = self.generate_sprites()

        t.log(f'Resolving action references')
        sprites = self.resolve_refs(sprites)

        t.log(f'Adding sounds')
        if self._sounds:
            sprites.append(SoundEffects(len(self._sounds)))
            for s in self._sounds.values():
                sprites.append((s,))

        t.log(f'Adding strings')
        sprites.extend(self.strings.get_actions())

        t.log(f'Enumerating {len(sprites)} real sprites')
        data_offset = 14
        next_sprite_id = 1
        for s in sprites:
            if isinstance(s, tuple):
                for x in s:
                    x.sprite_id = next_sprite_id
                next_sprite_id += 1
                s = s[0]
            data_offset += s.get_data_size() + 5

        with open(filename, 'wb') as f:
            t.log(f'Writing pseudo sprites')
            f.write(b'\x00\x00GRF\x82\x0d\x0a\x1a\x0a')  # file header
            f.write(struct.pack('<I', data_offset))
            f.write(b'\x00')  # compression(1)
            # f.write(b'\x04\x00\x00\x00')  # num(4)
            # f.write(b'\xFF')  # grf_type(1)
            # f.write(b'\xb0\x01\x00\x00')  # num + 0xff -> recoloursprites() (257 each)
            self._write_pseudo_sprite(f, b'\x02\x00\x00\x00')

            for s in sprites:
                if isinstance(s, tuple):
                    self._write_pseudo_sprite(f, s[0].get_data(), grf_type=0xfd)
                else:
                    self._write_pseudo_sprite(f, s.get_data(), grf_type=0xff)
            f.write(b'\x00\x00\x00\x00')

            t.log(f'Writing real sprites')
            for sl in sprites:
                if not isinstance(sl, tuple):
                    continue
                for s in sl:
                    f.write(s.get_real_data(self._sprite_encoder))

            f.write(b'\x00\x00\x00\x00')

        t.stop()
        self._sprite_encoder.print_time_report()

    def wrap(self, func):
        def wrapper(*args, **kw):
            obj = func(*args, **kw)
            self.add(obj)
            return obj
        return wrapper

    def bind(self, cls):

        @functools.wraps(cls)
        def wrapper(*args, **kw):
            gen = cls(*args, **kw)
            self.add(gen)
            return gen

        # Expose attributes on the wrapper
        for name in dir(cls):
            obj = getattr(cls, name)
            if not name.startswith('__'):
                setattr(wrapper, name, obj)

        return wrapper


class NewGRF(BaseNewGRF):
    def __init__(self, *, grfid, name, description, version=None, min_compatible_version=None, format_version=8, url=None):
        super().__init__()

        if isinstance(grfid, str):
            grfid = grfid.encode('utf-8')

        if len(grfid) != 4:
            raise ValueError(f'Expected 4 symbols for GRFID, found {len(grfid)}: {grfid}')

        props = {'INFO': {'PALS': b'D'}}

        if version is not None:
            assert isinstance(version, int), type(version)
            props['INFO']['VRSN'] = version

        if min_compatible_version is not None:
            assert isinstance(min_compatible_version, int), type(min_compatible_version)
            props['INFO']['MINV'] = min_compatible_version

        if url is not None:
            assert isinstance(url, str), type(url)
            props['INFO']['URL_'] = url

        self._props = props
        self._params = []
        self._cargo_table = None
        self.grfid = grfid
        self.name = name
        self.description = description
        self.format_version = format_version

    def set_cargo_table(self, cargo_list):
        self._cargo_table = {}
        for i, c in enumerate(cargo_list):
            self._cargo_table[to_bytes(c)] = i

    def map_cargo_labels(self, labels):
        if self._cargo_table is None:
            raise RuntimeError(f'`{self.__class__.__name__}.map_cargo_lables` requires cargo_table to be set')
        return bytes(self._cargo_table[to_bytes(l)] for l in labels)

    def generate_sprites(self):
        if self._params:
            self._props['INFO']['NPAR'] = bytes((len(self._params),))
            self._props['INFO']['PARA'] = {}
            for i, p in enumerate(self._params):
                self._props['INFO']['PARA'][i] = p

        res = [
            SetProperties(self._props),
            SetDescription(
                format_version=self.format_version,
                grfid=self.grfid,
                name=self.name,
                description=self.description
            )
        ]

        if self._cargo_table is not None:
            res.append(DefineMultiple(
                feature=GLOBAL_VAR,
                first_id=0,
                count=len(self._cargo_table),
                props={
                    'cargo_table': list(self._cargo_table.keys())
                }
            ))

        return res + super().generate_sprites()

    def add_bool_parameter(self, *, name, description, default=None):
        data = {
            'NAME': name,
            'DESC': description,
            'TYPE': b'\x01',
        }
        if default is not None:
            if not isinstance(default, bool):
                raise ValueError('Default value for bool parameter should be bool')
            data['DFLT'] = int(default)
        self._params.append(data)

    def add_int_parameter(self, *, name, description, default=None, limits=None, enum=None):
        data = {
            'NAME': name,
            'DESC': description,
            'TYPE': b'\0',
        }

        if default is not None:
            if not isinstance(default, int):
                raise ValueError('Default value int parameter should be int')
            data['DFLT'] = int(default)

        if limits is not None:
            if not isinstance(limits, tuple) or len(limits) != 2 or any(not isinstance(x, int) for x in limits):
                raise ValueError('Limits should be tuple of (int, int)')
            data['LIMI'] = struct.pack('<II', *limits)

        if enum is not None:
            if not all(isinstance(k, int) and isinstance(v, str) for k, v in enum.items()):
                raise ValueError('Enum should be a dict of {int: str}')

            data['VALU'] = enum

        self._params.append(data)


# Defines g.Class for newgrf binding
# EXPORT_CLASSES = [
#     Define, DefineMultiple, Action3, Action4, Map,
#     SpriteSet, BasicSpriteLayout, AdvancedSpriteLayout, Switch,
#     SetProperties, ReplaceOldSprites, ReplaceNewSprites, SetDescription,
#     Comment, ActionC, Label, Action10, SoundEffects, ImportSound, If,
# ]

# for cls in EXPORT_CLASSES:

#     def func(self, *args, _cls=cls, **kw):
#         obj = _cls(*args, **kw)
#         self.add(obj)
#         return obj

#     setattr(BaseNewGRF, cls.__name__, func)
