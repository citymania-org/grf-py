import grf
from .vehicle import Vehicle


class RoadVehicle(Vehicle):
    Flags = grf.RVFlags

    class Speed:
        def __init__(self, precise_value):
            self.precise_value = precise_value
            self.value = (self.precise_value + 2) // 4

        def __int__(self):
            return self.value

        def __str__(self):
            return str(self.value)

        def __repr__(self):
            return f'RoadVehicle.Speed({self.precise_value})'

    @staticmethod
    def kmhish(speed):
        return RoadVehicle.Speed(speed * 2)

    @staticmethod
    def mph(speed):
        # TODO doesn't show exact mph in the game
        return RoadVehicle.Speed((speed * 16 + 4) // 5)

    def __init__(self, *, id, name, max_speed, length=None, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None, **props):
        super().__init__(grf.RV, liveries=liveries, callbacks=callbacks, purchase_sprite=purchase_sprite, additional_text=additional_text)
        if length is not None:
            if not 1 <= length <= 8:
                raise ValueError('Invalid length (expected 1..8)')

            if 'shorten_by' in props:
                raise ValueError('Length and shorten_by properties are mutually exclusive')

        self.id = id
        self.name = name

        if isinstance(max_speed, self.Speed):
            self.max_speed = max_speed
        else:
            self.max_speed = self.Speed(max_speed * 4)
        self._props = props

        if length is not None:
            self._props['shorten_by'] = 8 - length

    def _set_callbacks(self, g):
        res = super()._set_callbacks(g)

        if self.max_speed.precise_value >= 0x400:
            # TODO only set speed property
            self.callbacks.change_properties = grf.Switch(
                ranges={
                    0x15: self.max_speed.value,
                },
                default=self.callbacks.graphics.default,
                code='extra_callback_info1_byte',
            )

        return res

    def get_sprites(self, g):
        res = self._gen_purchase_sprites()

        if self.liveries:
            layouts = []
            for i, l in enumerate(self.liveries):
                layouts.append(grf.GenericSpriteLayout(
                    ent1=(i,),
                    ent2=(i,),
                ))

            if len(self.liveries) > 1:

                self.callbacks.graphics.default = grf.Switch(
                    related_scope=True,
                    ranges=dict(enumerate(layouts)),
                    default=layouts[0],
                    code='cargo_subtype',
                )
            else:
                self.callbacks.graphics.default = layouts[0]

        res.extend(self._set_callbacks(g))

        # Liveries
        if self.liveries and len(self.liveries) > 1:
            self.callbacks.cargo_subtype_text = grf.Switch(
                ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
                default=0x400,
                code='cargo_subtype',
            )

        self.callbacks.set_flag_props(self._props)

        res.extend(self._gen_name_sprites(g, self.id))

        res.append(definition := grf.Define(
            feature=grf.RV,
            id=self.id,
            props={
                'sprite_id': 0xff,
                'precise_max_speed': min(self.max_speed.precise_value, 0xff),
                'max_speed': min(self.max_speed.value, 0xff),
                **self._props
            }
        ))
        if self.liveries:
            res.append(grf.Action1(
                feature=grf.RV,
                set_count=len(self.liveries),
                sprite_count=8,
            ))

            for l in self.liveries:
                res.extend(l['sprites'])

        res.append(self.callbacks.make_map_action(definition))
        return res
