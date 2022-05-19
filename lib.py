from nml.grfstrings import NewGRFString, default_lang

import grf

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


def fake_info_text(props):
    return '{}'.join('{BLACK}' + k + ': {GOLD}' + v for k, v in props.items())


def kmhishph(speed):
    return speed * 2

# TODO doesn't show exact mph in the game
def mph(speed):
    return (speed * 16 + 4) // 5


def make_cb_switches(callbacks, maps, layout):
    # TODO combine similar switches?
    out_maps = {}
    for k, cblist in maps.items():
        if not cblist: continue
        out_maps[k] = grf.VarAction2(
            ranges={0: layout, **cblist},
            default=layout,
            code='current_callback',
        )
    default = layout
    if callbacks:
        default = grf.VarAction2(
            ranges={0: layout, **callbacks},
            default=layout,
            code='current_callback',
        )
    return default, out_maps


class StringManager(grf.SpriteGenerator):
    def __init__(self):
        self.strings = []

    def add(self, string):
        string_id = len(self.strings)
        self.strings.append(grf_compile_string(string))
        return string_id

    def get_sprites(self, g):
        return [grf.Action4(
            feature=grf.TRAIN,
            offset=0xd000,
            is_generic_offset=True,
            strings=self.strings,
        )]


class RoadVehicle(grf.SpriteGenerator):
    def __init__(self, *, id, name, liveries, max_speed, additional_text=None, livery_refits=None, **props):
        for l in liveries:
            if 'name' not in l:
                raise ValueError(f'RoadVehicle livery is missing the name')
            sprites = l.get('sprites')
            if sprites is None:
                raise ValueError(f'RoadVehicle livery {l["name"]} is missing sprites')
            if len(sprites) != 8:
                raise ValueError(f'RoadVehicle livery expects 8 sprites, found {len(sprites)}')

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self.additional_text = additional_text
        self.liveries = liveries
        self.props = props

    def get_sprites(self, g):
        cb_flags = 0

        purchase_callbacks = {}
        callbacks = {}

        res = [
            grf.Action4(
                feature=grf.RV,
                offset=self.id,
                is_generic_offset=False,
                strings=[self.name.encode('utf-8')]
            ),
        ]

        if self.additional_text:
            string_id = 0xd000 + self.id

            purchase_callbacks[0x23] = g.strings.add(self.additional_text)

        if self.max_speed >= 0x400:
            callbacks[0x36] = purchase_callbacks[0x36] = grf.VarAction2(
                ranges={
                    0x15: self.max_speed // 4,
                },
                default=layout,
                code='var(16, 0, 255)',
            )

        # Liveries
        callbacks[0x19] = grf.VarAction2(
            ranges={i: g.strings.add(l['name']) for i, l in enumerate(self.liveries)},
            default=0x400,
            code='cargo_subtype',
        )
        cb_flags |= 0x20

        if cb_flags:
            self.props['cb_flags'] = self.props.get('cb_flags', 0) | cb_flags

        res.append(grf.Action0(
            feature=grf.RV,
            first_id=self.id,
            count=1,
            props={
                'sprite_id': 0xff,
                'precise_max_speed': min(self.max_speed, 0xff),
                'max_speed': min(self.max_speed // 4, 0xff),
                **self.props
            }
        ))
        res.append(grf.Action1(
            feature=grf.RV,
            set_count=len(self.liveries),
            sprite_count=8,
        ))

        for l in self.liveries:
            res.extend(l['sprites'])

        layouts = []
        for i, l in enumerate(self.liveries):
            layouts.append(grf.GenericSpriteLayout(
                ent1=(i,),
                ent2=(i,),
            ))

        layout = grf.VarAction2(
            related_scope=True,
            ranges=dict(enumerate(layouts)),
            default=layouts[0],
            code='cargo_subtype',
        )

        default, maps = make_cb_switches(callbacks, {255: purchase_callbacks}, layout)
        res.append(grf.Action3(
            feature=grf.RV,
            ids=[self.id],
            maps=maps,
            default=default,
        ))
        return res


class Train(grf.SpriteGenerator):
    class EngineClass:
        STEAM = 0x0
        DIESEL = 0x8
        ELECTRIC = 0x28
        MONORAIL = 0x32
        MAGLEV = 0x32


    def __init__(self, *, id, name, liveries, max_speed, additional_text=None, livery_refits=None, **props):
        for l in liveries:
            if 'name' not in l:
                raise ValueError(f'Train livery is missing the name')
            sprites = l.get('sprites')
            if sprites is None:
                raise ValueError(f'Train livery {l["name"]} is missing sprites')
            if len(sprites) != 8:
                raise ValueError(f'Train livery expects 8 sprites, found {len(sprites)}')

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self.additional_text = additional_text
        self.liveries = liveries
        REQUIRED_PROPS = ('engine_class', )
        missing_props = [p for p in REQUIRED_PROPS if p not in props]
        if missing_props:
            raise ValueError('Missing required properties for Train: {}'.format(', '.join(missing_props)))
        self._props = props
        self._articulated_parts = []

    def add_articulated_part(self, *, id, liveries, **props):
        if self._props.get('is_dual_headed'):
            raise RuntimeError('Articulated parts are not allowed for dual-headed engines')

        # REQUIRED_PROPS = ('id', 'liveries')
        # missing_props = [p for p in REQUIRED_PROPS if p not in props]
        # if missing_props:
        #     raise ValueError('Articulated part is missing required property: {}'.format(', '.join(missing_props)))

        ALLOWED_PROPS = (
            'cargo_capacity',
            'default_cargo_type',
            'refit_cost',
            'refittable_cargo_types',
            'shorten_by',
            'visual_effect',
            'bitmask_vehicle_info',
            'misc_flags',
            'refittable_cargo_classes',
            'non_refittable_cargo_classes',
            'cargo_age_period',
            'cargo_allow_refit',
            'cargo_disallow_refit',
            'curve_speed_mod',
        )
        invalid_props = [p for p in props if p not in ALLOWED_PROPS]
        if invalid_props:
            raise ValueError('Property not allowed for articulated part: {}'.format(', '.join(invalid_props)))

        self._articulated_parts.append((id, liveries, props))
        return self

    def get_sprites(self, g):
        # Check in case property was changed after add_articulated
        if self._props.get('is_dual_headed') and self._articulated_parts:
            raise RuntimeError('Articulated parts are not allowed for dual-headed engines (vehicle id {self.id})')

        cb_flags = 0

        purchase_callbacks = {}
        callbacks = {}

        res = [
            grf.Action4(
                feature=grf.TRAIN,
                offset=self.id,
                is_generic_offset=False,
                strings=[self.name.encode('utf-8')]
            ),
        ]

        if self.additional_text:
            string_id = 0xd000 + self.id

            purchase_callbacks[0x23] = g.strings.add(self.additional_text)

        # Liveries
        callbacks[0x19] = grf.VarAction2(
            ranges={i: g.strings.add(l['name']) for i, l in enumerate(self.liveries)},
            default=0x400,
            code='cargo_subtype',
        )
        cb_flags |= 0x20

        if self._articulated_parts:
            callbacks[0x16] = grf.VarAction2(
                ranges={i + 1: ap[0] for i, ap in enumerate(self._articulated_parts)},
                default=0x7fff,
                code='extra_callback_info1 & 255',
            )
            cb_flags |= 0x10


        if cb_flags:
            self._props['cb_flags'] = self._props.get('cb_flags', 0) | cb_flags

        res.append(grf.Action0(
            feature=grf.TRAIN,
            first_id=self.id,
            count=1,
            props={
                'sprite_id': 0xfd,  # magic value for newgrf sprites
                'max_speed': self.max_speed // 2,
                **self._props
            }
        ))
        res.append(grf.Action1(
            feature=grf.TRAIN,
            set_count=len(self.liveries),
            sprite_count=8,
        ))

        for l in self.liveries:
            res.extend(l['sprites'])

        layouts = []
        for i, l in enumerate(self.liveries):
            layouts.append(grf.GenericSpriteLayout(
                ent1=(i,),
                ent2=(i,),
            ))

        layout = grf.VarAction2(
            related_scope=True,
            ranges=dict(enumerate(layouts)),
            default=layouts[0],
            code='cargo_subtype',
        )

        default, maps = make_cb_switches(callbacks, {255: purchase_callbacks}, layout)
        res.append(grf.Action3(
            feature=grf.TRAIN,
            ids=[self.id],
            maps=maps,
            default=default,
        ))

        for apid, liveries, props in self._articulated_parts:
            res.append(grf.Action0(
                feature=grf.TRAIN,
                first_id=apid,
                count=1,
                props={
                    'sprite_id': 0xfd,  # magic value for newgrf sprites
                    'engine_class': self._props.get('engine_class'),
                    **props
                }
            ))

            res.append(grf.Action1(
                feature=grf.TRAIN,
                set_count=len(liveries),
                sprite_count=8,
            ))

            layouts = []
            for i, l in enumerate(liveries):
                res.extend(l['sprites'])
                layouts.append(grf.GenericSpriteLayout(
                    ent1=(i,),
                    ent2=(i,),
                ))

            layout = grf.VarAction2(
                related_scope=True,
                ranges=dict(enumerate(layouts)),
                default=layouts[0],
                code='cargo_subtype',
            )

            res.append(grf.Action3(
                feature=grf.TRAIN,
                ids=[apid],
                maps={},
                default=layout,
            ))
        return res
