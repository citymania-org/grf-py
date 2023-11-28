import grf
from .vehicle import Vehicle
from .callback import make_callback_manager


class Train(Vehicle):
    Flags = grf.TrainFlags
    AIFlags = grf.AIFlags
    RunningCost = grf.TrainRunningCost
    VisualEffect = grf.TrainVisualEffect

    class EngineClass:
        STEAM = 0x0
        DIESEL = 0x8
        ELECTRIC = 0x28
        MONORAIL = 0x32
        MAGLEV = 0x32

    @staticmethod
    def kmhish(speed):
        return speed

    @staticmethod
    # TODO doesn't show exact mph in the game
    def mph(speed):
        return (speed * 16 + 4) // 10

    @staticmethod
    def hp(value):
        return int(value + .5)

    @staticmethod
    def ton(value):
        return int(value + .5)

    @staticmethod
    def visual_effect_and_powered(effect, *, position=0, wagon_power=True):
        return effect | position | wagon_power * 0x7f

    def __init__(self, *, id, name, max_speed, weight, length=None, sound_effects=None, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None, **props):
        super().__init__(grf.TRAIN, liveries=liveries, callbacks=callbacks, purchase_sprite=purchase_sprite, additional_text=additional_text)

        REQUIRED_PROPS = ('engine_class', )
        missing_props = [p for p in REQUIRED_PROPS if p not in props]
        if missing_props:
            raise ValueError('Missing required properties for Train: {}'.format(', '.join(missing_props)))

        if length is not None and 'shorten_by' in props:
            raise ValueError('Length and shorten_by properties are mutually exclusive')

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self.sound_effects = sound_effects
        self.weight = weight
        self.length = length
        self._props = props
        self._articulated_parts = []

        mid_shorten, art_shorten, art_liveries = self._calc_length_articulation(
            length, self._props.get('shorten_by'), self.liveries)
        if art_shorten is not None:
            self._head_liveries = art_liveries
            self._props['shorten_by'] = art_shorten
            self._add_auto_articulated_parts(self.id, mid_shorten, self.liveries, art_shorten, art_liveries, self._props)
        else:
            self._head_liveries = self.liveries
            if mid_shorten is not None:
                self._props['shorten_by'] = mid_shorten

    def _add_auto_articulated_parts(self, id, mid_shorten, mid_liveries, art_shorten, art_liveries, props):
        # TODO move auto-articulated stuff to the generation phase so props can be changed after creation.
        art_props = {}
        flags = self._props.get('misc_flags')
        if flags is not None:
            copy_flags = self.Flags.TILT | self.Flags.MULTIPLE_UNIT | self.Flags.USE_2CC
            art_props['misc_flags'] = flags & copy_flags
        # Copy power for the right 2cc colour, effective power will be zero anyway.
        for k in ('track_type', 'power'):
            if k in props:
                art_props[k] = props[k]
        self._do_add_articulated_part(f'__{id}_aa_mid', mid_shorten, mid_liveries, mid_liveries, art_props)
        self._do_add_articulated_part(f'__{id}_aa_tail', art_shorten, art_liveries, mid_liveries, art_props)

    def _calc_length_articulation(self, length, shorten_by, liveries):
        if length is None:
            return shorten_by, None, None

        if length > 24:
            raise ValueError("Max length of Train part is 24")

        if length <= 8:
            return 8 - length, None, None

        if self.length % 2 == 1:
            central_length = 7
            articulated_length = (length - 7) // 2
        else:
            central_length = 8
            articulated_length = (length - 8) // 2
        articulated_liveries = [{
            'name': l['name'],
            'sprites': [grf.EMPTY_SPRITE] * 8,
        } for l in liveries or []]

        return 8 - central_length, 8 - articulated_length, articulated_liveries

    def _do_add_articulated_part(self, id, shorten_by, liveries, mid_liveries, props, callbacks=None):
        if shorten_by is not None:
            props = {
                **props,
                'shorten_by': shorten_by,
            }
        self._articulated_parts.append((id, liveries, mid_liveries, callbacks, props))

    def add_articulated_part(self, *, id, liveries=None, callbacks=None, skip_props_check=False, length=None, **props):
        if not skip_props_check:
            # TODO support flipping (id + 0x4000)
            if isinstance(id, int) and id >= 0x4000:
                raise ValueError(f'Articulated part id too big ({id} >= 0x4000)')
            if self._props.get('is_dual_headed'):
                raise RuntimeError('Articulated parts are not allowed for dual-headed engines')

            if length is not None and 'shorten_by' in props:
                raise ValueError('Length and shorten_by properties are mutually exclusive')

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
                'visual_effect_and_powered',
                'bitmask_vehicle_info',
                'misc_flags',
                'refittable_cargo_classes',
                'non_refittable_cargo_classes',
                'cargo_age_period',
                'cargo_allow_refit',
                'cargo_disallow_refit',
                'curve_speed_mod',
                'power',  # Game will zero power but it's needed for the 2cc colour scheme
            )
            invalid_props = [p for p in props if p not in ALLOWED_PROPS]
            if invalid_props:
                raise ValueError('Property not allowed for articulated part: {}'.format(', '.join(invalid_props)))

        mid_shorten, art_shorten, art_liveries = self._calc_length_articulation(
            length, props.get('shorten_by'), liveries)

        if art_shorten is None:
            self._do_add_articulated_part(id, mid_shorten, liveries, liveries, props, callbacks)
        else:
            self._do_add_articulated_part(id, art_shorten, art_liveries, liveries, props, callbacks)
            self._add_auto_articulated_parts(id, mid_shorten, liveries, art_shorten, art_liveries, props)

        return self

    def _make_graphics(self, liveries, position):
        assert liveries
        res = [grf.Action1(
            feature=grf.TRAIN,
            set_count=len(liveries),
            sprite_count=8,
        )]
        layouts = []
        for i, l in enumerate(liveries):
            res.extend(l['sprites'])
            layouts.append(grf.GenericSpriteLayout(
                ent1=(i,),
                ent2=(i,),
            ))

        if len(liveries) <= 1:
            return res, layouts[0]

        subtype_code = 'cargo_subtype'
        if position > 0:
            subtype_code=f'''
                TEMP[0x10F]=-{position}
                var(0x61, param=0xF2, shift=0, and=0xFF)
            ''',

        return res, grf.Switch(
            related_scope=False,
            ranges=dict(enumerate(layouts)),
            default=layouts[0],
            code=subtype_code,
        )

    def _set_callbacks(self, g):
        res = super()._set_callbacks(g)

        if self.liveries and len(self.liveries) > 1:
            # Liveries
            self.callbacks.cargo_subtype_text = grf.Switch(
                ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
                default=0x400,
                code='cargo_subtype',
            )

        if self.sound_effects:
            self.callbacks.sound_effect = grf.Switch(
                ranges=self.sound_effects,
                default=self.callbacks.graphics.default,
                code='extra_callback_info1_byte',
            )

        if self._articulated_parts:
            self.callbacks.articulated_part = grf.Switch(
                ranges={i + 1: g.resolve_id(self.feature, ap[0], articulated=True) for i, ap in enumerate(self._articulated_parts)},
                default=0x7fff,
                code='extra_callback_info1_byte',
            )
        return res

    def _set_articulated_part_callbacks(self, g, position, callbacks):
        pass


    def _gen_purchase_sprites(self):
        if self.purchase_sprite is None and (self.length or 8) > 8 and self.liveries is not None:
            # First part has no graphics so set purchase sprite
            self.purchase_sprite = self.liveries[0]['sprites'][6]
        return super()._gen_purchase_sprites()


    def get_sprites(self, g):
        # Check in case property was changed after add_articulated
        if self._props.get('is_dual_headed') and self._articulated_parts:
            raise RuntimeError('Articulated parts are not allowed for dual-headed engines (vehicle id {self.id})')

        res = self._gen_purchase_sprites()

        if self._head_liveries:
            sprites, self.callbacks.graphics.default = self._make_graphics(self._head_liveries, 0)
            res.extend(sprites)

        res.extend(self._set_callbacks(g))

        self.callbacks.set_flag_props(self._props)

        tid = g.resolve_id(self.feature, self.id)
        res.extend(self._gen_name_sprites(g, tid))

        res.append(definition := grf.Define(
            feature=grf.TRAIN,
            id=tid,
            props={
                'sprite_id': 0xfd,  # magic value for newgrf sprites
                'max_speed': self.max_speed,
                'weight_low': self.weight % 256,
                'weight_high': self.weight // 256,
                **self._props
            }
        ))

        res.extend(self.callbacks.make_map_action(definition))

        position = 0
        for apid, liveries, _, initial_callbacks, props in self._articulated_parts:
            apid = g.resolve_id(self.feature, apid, articulated=True)
            position += 1
            callbacks = make_callback_manager(self.feature, initial_callbacks)
            self._set_articulated_part_callbacks(g, position, callbacks)

            if liveries:
                sprites, callbacks.graphics.default = self._make_graphics(liveries, position)
                res.extend(sprites)

            callbacks.set_flag_props(props)

            res.append(definition := grf.Define(
                feature=grf.TRAIN,
                id=apid,
                props={
                    'sprite_id': 0xfd,  # magic value for newgrf sprites
                    'engine_class': self._props.get('engine_class'),
                    **props
                }
            ))
            res.extend(callbacks.make_map_action(definition))

        return res


class SetGlobalTrainDepotYOffset(grf.SpriteGenerator):
    def __init__(self, offset):
        self.offset = offset

    def get_sprites(self, g):
        return [
            grf.ComputeParameters(
                target=0x8e,
                operation=0x00,
                if_undefined=False,
                source1=0xff,
                source2=0xff,
                value=self.offset,
            )
        ]


class GlobalTrainMiscFlag:
    DESERT_TREES_AND_FIELDS = 0
    DESERT_PAVEMENT_AND_LIGHTS = 1
    FIELDS_BOUNDING_BOX = 2
    DEPOT_FULL_TRAIN_WIDTH = 3
    AMBIENT_SOUND_CB = 4
    THIRD_TRACK_CATENARIES = 5
    SECOND_ROCKY_SET = 6
    TTDPATCH_FLAG = 31


class SetGlobalTrainMiscFlag(grf.SpriteGenerator):
    def __init__(self, flag):
        self.flag = flag

    def get_sprites(self, g):
        return [
            grf.ComputeParameters(
                target=0x9e,
                operation=0x08,
                if_undefined=False,
                source1=0x9e,
                source2=0xff,
                value=1 << self.flag
            )
        ]
