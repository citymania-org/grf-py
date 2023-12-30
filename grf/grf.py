import functools
import heapq
import inspect
import json
import os
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
                     ModifySprites, If, SetDescription, ReplaceOldSprites, ErrorMessage, Comment, \
                     ComputeParameters, Label, SoundEffects, ImportSound, Translations, SetProperties
from .parser import Node, Expr, Value, Var, Temp, Perm, Call, parse_code, OP_INIT, SPRITE_FLAGS, GenericVar
from .common import Feature, hex_str, utoi32, FeatureMeta, to_bytes, GLOBAL_VAR
from .common import INDUSTRY_TILE, INDUSTRY
from .colour import PIL_PALETTE
from .cache import SpriteCache
from .sprites import Action, Sprite, Sound, ResourceAction, FakeAction, Resource, \
                     PaletteRemap, AlternativeSprites, ResourceFile, LoadedResourceFile, \
                     SingleResourceAction
from .strings import StringManager, StringRef


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
        im.putpalette(PIL_PALETTE)

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


class DummySprite(Action):
    def get_data(self):
        return b'\x00'


class SpriteGenerator:
    def get_ (self, grf):
        return NotImplementedError


class PythonGenerationContext:
    def __init__(self):
        self.resources = {}
        self.use_vars_not_refs = True
        self._var_count = {}


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
        self.num_sprites = 0
        self.num_cached = 0
        self.num_uncacheable = 0
        self.num_duplicate = 0

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
        print(f'Total {self.num_sprites} sprites, cached {self.num_cached}, non-cacheable {self.num_uncacheable}. Optimized {self.num_duplicate} duplicates.')


class BaseNewGRF:
    def __init__(self, *, strings=None, id_map_file=None, sprite_cache_path='.cache'):
        self.generators = []
        self._next_sound_id = 73
        self._sounds = {}
        self.strings = StringManager() if strings is None else strings
        self._sprite_encoder = SpriteEncoder()
        self._id_map = IDMap(id_map_file)
        self.sprite_cache_path = sprite_cache_path

    def reserve_ids(self, feature, ids):
        self._id_map.reserve_ids(feature, ids)

    def resolve_id(self, feature, value, *, articulated=False):
        return self._id_map.resolve(feature, value, articulated=articulated)

    def _add_sound(self, s):
        assert isinstance(s, Sound)

        h = s.get_fingerprint()
        if h in self._sounds:
            s.id = self._sounds[h].resource.id
        else:
            if self._next_sound_id > 0xff:
                raise RuntimeError('Too many sound effects (max 183)')
            s.id = self._next_sound_id
            self._next_sound_id += 1
            self._sounds[h] = SingleResourceAction(s)

    def _add(self, l, sprite):
        # TODO assert all(isinstance(s, (BaseSprite, SpriteGenerator, PyComment)) for s in sprites), sprites

        if isinstance(sprite, SingleResourceAction):
            if isinstance(sprite.resource, Sound):
                self._add_sound(sprite)
            return

        if isinstance(sprite, Sound):
            self._add_sound(sprite)
            return

        if isinstance(sprite, Resource):
            sprite = SingleResourceAction(sprite)

        l.append(sprite)

    def add(self, *sprites):
        for s in sprites:
            self._add(self.generators, s)

    def _write_pseudo_sprite(self, f, data, grf_type=0xff):
        f.write(struct.pack('<IB', len(data), grf_type))
        res = f.tell()
        f.write(data)
        return res

    def generate_python(self, grf_filename):
        context = PythonGenerationContext()
        res = 'import datetime\nimport grf\n'
        make_grf_import = lambda l: 'from grf import ' + ', '.join(l) + '\n'
        res += make_grf_import(f.constant for f in FeatureMeta.FEATURES)
        res += 'from grf import ZOOM_4X, ZOOM_NORMAL, ZOOM_2X, ZOOM_8X, ZOOM_16X, ZOOM_32X\n'
        res += 'from grf import Ref, CB, ImageFile, FileSprite, FileMask, RAWSound, PaletteRemap\n\n'
        res += 'g = grf.BaseNewGRF()\n\n'
        used_classes = set(x.__class__.__name__ for x in self.generators if not isinstance(x, (tuple, FakeAction)))
        # used_classes.update(('ImageFile', 'ImageSprite', 'RAWSound'))  # TODO check usage of FakeAction
        if used_classes:
            res += '# Bind all the used classes to current grf so they can be used declaratively\n'
            res += ', '.join(used_classes) + ' = ' +  ', '.join(f'g.bind(grf.{c})' for c in used_classes)
            res += '\n'
        res += '\n'

        refs = {}
        ref_count = {}
        for s in self.generators:
            if isinstance(s, ReferencingAction):
                new_refs = []
                for r in s.get_refs():
                    if isinstance(r, Ref) and not r.is_callback:
                        nr = refs.get(r.value)
                        if nr is not None:
                            new_refs.append(nr)
                            continue
                    new_refs.append(r)
                s.set_refs(new_refs)
            if isinstance(s, ReferenceableAction) and s.ref_id is not None:
                refs[s.ref_id] = s
                count = ref_count.get(s.ref_id, 0) + 1
                ref_count[s.ref_id] = count
                s.ref_var = f'ref_{s.ref_id}_{count}'

        for s in self.generators:
            if isinstance(s, tuple):
                s = s[0]
            code = textwrap.dedent(s.py(context)).strip()
            if isinstance(s, ReferenceableAction) and s.ref_var is not None:
                res += f'{s.ref_var} = '
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
                        robj = refids.get(r.ref_id)
                        if robj is None:
                            print (f'WARNING Unresolved direct reference {r} in action {s.py(None)}')
                            continue
                        r = robj
                        ref_count[id(r)] += 1
                        continue

                    if not isinstance(r, ReferenceableAction):
                        if isinstance(r, Sound):
                            self._add_sound(r)
                        continue

                    if r is s:
                        raise RuntimeError(f'Action {s} references itself')

                    assert isinstance(r, ReferenceableAction)

                    ref_count[id(r)] += 1
                    if id(r) not in actions:
                        actions[id(r)] = (r, None)
                        unordered_refs[id(s)].append(id(r))
                        resolve(r, None)
                    else:
                        if id(r) not in ordered_refs[id(s)]:
                            ordered_refs[id(s)].append(id(r))

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

    def _enumerate_sprites(self, sprites):
        # We want repeatable result so can't check sprite cache for better ordering
        next_sprite_id = 1
        unordered_sprites = []
        ordered_sprites = []
        fingerprints = {}
        for s in sprites:
            if isinstance(s, ResourceAction):
                resource_files = s.get_resource_files()
                for f in resource_files:
                    assert isinstance(f, ResourceFile), type(f)
                loaded_resources = [f for f in resource_files if isinstance(f, LoadedResourceFile)]
                if loaded_resources:
                    # We'll assign them later
                    unordered_sprites.append((s, loaded_resources))
                    continue
                s.sprite_id = next_sprite_id
                next_sprite_id += 1
                ordered_sprites.append(s)

        # Try to order sprites in a way that minimizes memory consumption for loaded images

        img_index = defaultdict(list)
        sprite_priority = {}
        sprite_idx = {}
        q = []
        for s, files in unordered_sprites:
            for f in files:
                img_index[id(f)].append(id(s))
            sprite_idx[id(s)] = (s, files)
            p = len(files)
            sprite_priority[id(s)] = p
            heapq.heappush(q, (p, id(s)))

        load_files = defaultdict(list)
        last_use = {}
        processed_sprites = set()
        while q:
            _, sid = heapq.heappop(q)
            if sid in processed_sprites:
                continue
            processed_sprites.add(sid)
            s, files = sprite_idx[sid]
            ordered_sprites.append(s)
            s.sprite_id = next_sprite_id
            next_sprite_id += 1
            for f in files:
                fid = id(f)
                if fid not in last_use:
                    load_files[sid].append(f)
                last_use[fid] = (sid, f)
                for x in img_index[fid]:
                    sprite_priority[x] -= 1
                    heapq.heappush(q, (sprite_priority[x], x))

        unload_files = defaultdict(list)
        for sid, f in last_use.values():
            unload_files[sid].append(f)

        res = []
        for s in ordered_sprites:
            res.append((s, load_files.get(id(s)), unload_files.get(id(s))))

        return res

    def write(self, filename, clean_build=False, debug_zoom_levels=False):
        t = Timer()
        t.start(f'Evaluating sprite generators')
        sprites = self.generate_sprites()

        t.log(f'Resolving action references')
        sprites = self.resolve_refs(sprites)

        t.log(f'Adding sounds')
        if self._sounds:
            sprites.append(SoundEffects(len(self._sounds)))
            sprites.extend(self._sounds.values())

        t.log(f'Adding strings')
        sprites = self.strings.get_persistent_actions() + sprites + self.strings.get_actions()

        t.log(f'Enumerating {len(sprites)} real sprites')
        sprite_cache = SpriteCache(self.sprite_cache_path)
        sprite_cache.load(clean_build=clean_build)
        sprite_order = self._enumerate_sprites(sprites)

        t.log(f'Writing actions')
        with open(filename, 'wb') as f:
            f.write(b'\x00\x00GRF\x82\x0d\x0a\x1a\x0a')  # file header
            data_offset_pos = f.tell()
            f.write(b'\x00\x00\x00\x00')  # data offset (overwritten later)
            f.write(b'\x00')  # compression(1)
            # f.write(b'\x04\x00\x00\x00')  # num(4)
            # f.write(b'\xFF')  # grf_type(1)
            # f.write(b'\xb0\x01\x00\x00')  # num + 0xff -> recoloursprites() (257 each)
            self._write_pseudo_sprite(f, b'\x02\x00\x00\x00')

            # Using dict of lists as each sprite can be used multiple times
            resource_references = defaultdict(list)
            for s in sprites:
                if isinstance(s, ResourceAction):
                    resource_references[s.sprite_id].append(self._write_pseudo_sprite(f, s.get_data(), grf_type=0xfd))
                else:
                    self._write_pseudo_sprite(f, s.get_data(), grf_type=0xff)

            data_offset = f.tell() - data_offset_pos
            f.write(b'\x00\x00\x00\x00')

            t.log(f'Writing resources')
            file_mod_date = {}

            def get_sprite_fingerprint(s):
                if not isinstance(s, Sprite):
                    return None

                fingerprint = s.get_fingerprint()
                if fingerprint is None:
                    return None

                files = s.get_resource_files()
                files_data = {}
                for f in files:
                    path = str(f.path)
                    fmod = file_mod_date.get(path)
                    if fmod is None:
                        fmod = file_mod_date[path] = os.path.getmtime(path)
                    files_data[path] = fmod

                return {
                    'data': fingerprint,
                    'files': files_data,
                }

            def get_sprite_data(s):
                sprite_data = None
                if debug_zoom_levels:
                    s = ZoomDebugRecolourSprite(s)
                if isinstance(s, Sprite):
                    sprite_data = get_sprite_fingerprint(s)
                if sprite_data is not None:
                    data = sprite_cache.get(sprite_data)
                    if data is None:
                        data = s.get_real_data(self._sprite_encoder)
                        sprite_cache.set(sprite_data, data)
                    else:
                        self._sprite_encoder.num_cached += 1
                else:
                    data = s.get_real_data(self._sprite_encoder)
                    self._sprite_encoder.num_uncacheable += 1

                self._sprite_encoder.num_sprites += 1
                return data

            renumerate_sprites = {}
            data_hashes = {}
            written_resources = set()
            for sl, load_files, unload_files in sprite_order:
                if load_files:
                    for rf in load_files:
                        rf.load()

                if isinstance(sl, ResourceAction):
                    if sl in written_resources:
                        continue

                    data = []
                    for s in sl.get_resources():
                        d = get_sprite_data(s)
                        data.append(d)

                    h = hash(tuple(data))
                    sid = data_hashes.get(h)
                    if sid is not None:
                        # Don't duplicate sprite, just change the reference.
                        renumerate_sprites[sl.sprite_id] = sid
                    else:
                        # Unique sprite, write it and add to index.
                        for d in data:
                            f.write(struct.pack('<II', sl.sprite_id, len(d)))
                            f.write(d)
                        data_hashes[h] = sl.sprite_id

                    written_resources.add(sl)


                if unload_files:
                    for rf in unload_files:
                        rf.unload()

            f.write(b'\x00\x00\x00\x00')

            f.seek(data_offset_pos)
            f.write(struct.pack('<I', data_offset))

            for k, v in renumerate_sprites.items():
                for ofs in resource_references[k]:
                    f.seek(ofs)
                    f.write(struct.pack('<I', v))

            self._sprite_encoder.num_duplicate = len(renumerate_sprites)

        sprite_cache.save()
        self._id_map.save()
        t.stop()
        self._sprite_encoder.print_time_report()

        watched = set()
        for s in sprites:
            if isinstance(s, ResourceAction):
                for f in s.get_resource_files():
                    assert isinstance(f, ResourceFile)
                    watched.add(f.path)

        return watched

    def wrap(self, func):
        def wrapper(*args, **kw):
            obj = func(*args, **kw)
            self.add(obj)
            return obj
        return wrapper

    def bind(self, cls):
        def constructor(obj_self, *args, **kw):
            cls.__init__(obj_self, *args, **kw)
            self.add(obj_self)

        return type(
            'Bound' + cls.__name__,
            (cls,),
            {
                'bound_newgrf': self,
                '__init__': constructor,
            },
        )


class IDMap:
    MIN_ID = 300

    def __init__(self, path=None):
        self.path = path
        self._loaded = False
        self._index = {}
        self._next_id = {}
        self._auto_ids = set()
        self._manual_ids = set()

    def _check_path(self):
        if self.path is None:
            self.path = 'id_map.json'
            print(f'WARNING: ID index is not configured, using default "{self.path}"')
            print('Set id index path by passing id_map_file to NewGRF constructor.')

    def _load(self):
        if self._loaded: return
        self._check_path()

        try:
            data = json.load(open(self.path))
        except FileNotFoundError:
            # TODO link docs on how to create empty index
            # Creating index automatically is dangerous as it may mask the error.
            raise RuntimeError(f'ERROR: ID index file is not found in "{self.path}".')

        assert data['version'] == 1

        # Load into temporaries first in case of exception
        index = {}
        next_id = {}
        used_ids = set()
        for feature_name, feature_index in data['index'].items():
            feature = Feature.from_name(feature_name)
            for name, fid in feature_index.items():
                key = (feature, name)
                index[key] = fid
                used_ids.add(key)
                next_id[feature] = max(next_id.get(feature, 0), fid + 1)
        self._loaded = True
        self._index = index
        self._next_id = next_id
        self._auto_ids = used_ids

    def save(self):
        if not self._index:
            return
        self._check_path()
        index = {}
        for (feature, name), value in self._index.items():
            index.setdefault(feature.name, {})[name] = value
        data = {
            'version': 1,
            'index': index,
        }
        jdata = json.dumps(data, indent=4)
        with open(self.path, 'w') as f:
            f.write(jdata)

    def reserve_ids(self, ids):
        self._manual_ids.update(ids)

    def _get_min_id(self, feature):
        if feature == INDUSTRY_TILE:
            return 0
        if feature == INDUSTRY:
            return 37
        return self.MIN_ID

    def resolve(self, feature, value, *, articulated=False):
        self._load()
        key = (feature, value)
        if isinstance(value, int):
            if key in self._auto_ids:
                raise RuntimeError(f'ID {value} can''t be used for feature {feature!r} as it was previously auto-assigned.')
            self._manual_ids.add(key)
            return value

        assert isinstance(value, str), type(value)
        i = self._index.get(key)
        # TODO use separate articulated range
        if i is None:
            i = self._next_id.get(feature, self._get_min_id(feature))
            while (feature, i) in self._manual_ids:
                i += 1
            self._next_id[feature] = i + 1
            self._index[key] = i
            self._auto_ids.add((feature, i))
        return i


class NewGRF(BaseNewGRF):
    BLITTER_BPP_8 = b'8'
    BLITTER_BPP_32 = b'3'

    def __init__(self, *, grfid, name, description, version=None, min_compatible_version=None, format_version=8, url=None, strings=None, id_map_file=None, sprite_cache_path='.cache', preferred_blitter=None):
        super().__init__(strings=strings, id_map_file=id_map_file, sprite_cache_path=sprite_cache_path)

        if isinstance(grfid, str):
            grfid = grfid.encode('utf-8')

        if len(grfid) != 4:
            raise ValueError(f'Expected 4 symbols for GRFID, found {len(grfid)}: {grfid}')

        props = {
            'INFO': {
                'PALS': b'D',
                'NAME': name,
                'DESC': description,
            }
        }

        if preferred_blitter is not None:
            if preferred_blitter not in (NewGRF.BLITTER_BPP_8, NewGRF.BLITTER_BPP_32):
                raise ValueError(f'Invalid value for preferred_blitter: {preferred_blitter}')
            props['INFO']['BLTR'] = preferred_blitter

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
        self._railtype_table = None
        self.grfid = grfid
        self.name = name
        self.description = description
        self.format_version = format_version

    @property
    def grfid_value(self):
        return struct.unpack('<I', self.grfid)[0]

    def add_railtype(self, *railtype_list):
        if self._railtype_table is None:
            self._railtype_table = {}

        if railtype_list and isinstance(railtype_list[0], (tuple, list)):
            railtype_list = railtype_list[0]

        rt_id = len(self._railtype_table)

        if isinstance(railtype_list, (list, tuple)):
            rtb = to_bytes(railtype_list[0])
            self._railtype_table[rtb] = (rt_id, tuple(map(to_bytes, railtype_list)))
        else:
            rtb = to_bytes(railtype_list)
            self._railtype_table[rtb] = (rt_id, None)

        return rt_id

    def set_railtype_table(self, railtype_list):
        self._railtype_table = {}
        res = []
        for i, rt in enumerate(railtype_list):
            res.append(self.add_railtype(rt))
        return res

    def get_railtype_id(self, railtype):
        rtbytes = to_bytes(railtype)
        if rtbytes not in self._railtype_table:
            raise ValueError(f'Unknown railtype `{railtype}`')
        return self._railtype_table[rtbytes][0]

    def set_cargo_table(self, cargo_list):
        self._cargo_table = {}
        for i, c in enumerate(cargo_list):
            self._cargo_table[to_bytes(c)] = i
        return tuple(range(len(cargo_list)))

    def get_cargo_id(self, cargo):
        cbytes = to_bytes(cargo)
        res = self._cargo_table.get(cbytes)
        assert res is not None, cbytes
        return res

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
                name=str(self.name),
                description=str(self.description),
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

        res.extend(self._generate_railtype_table())

        return res + super().generate_sprites()

    def _generate_railtype_table(self):
        if self._railtype_table is None:
            return []

        res = []

        mods = []
        for i, rtlist in self._railtype_table.values():
            if rtlist is None or len(rtlist) == 1:
                continue

            param = 0x7f - len(mods)
            for j, rt in enumerate(rtlist):
                # [param] = rt
                res.append(ComputeParameters(
                    target=param,
                    operation=0,
                    if_undefined=False,
                    source1=255,
                    source2=255,
                    value=rt,
                ))

                # if rt is defined skip the rest (unless it's the last rt)
                skip = 2 * (len(rtlist) - j - 1) - 1
                if skip > 0:
                    res.append(If(
                        is_static=True,
                        variable=0,
                        condition=0xe,
                        value=rt,
                        skip=skip,
                    ))

            mods.append({'num': param, 'size': 4, 'offset': 8 + i * 4})

        if mods:
            res.append(ModifySprites(mods))

        res.append(DefineMultiple(
            feature=GLOBAL_VAR,
            first_id=0,
            count=len(self._railtype_table),
            props={
                'railtype_table': list(self._railtype_table.keys())
            }
        ))
        return res

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
            if not all(isinstance(k, int) and isinstance(v, (str, StringRef)) for k, v in enum.items()):
                raise ValueError('Enum should be a dict of {int: Union[str, StringRef]}')

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
