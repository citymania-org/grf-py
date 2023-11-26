from collections import defaultdict


import grf
from .callback import make_callback_manager


class SpriteTable(grf.SpriteGenerator):
    def __init__(self, feature):
        assert isinstance(feature, grf.Feature)
        self.feature = feature
        self._rows = []
        self._width = None

    def add_row(self, row):
        assert isinstance(row, (list, tuple))
        if self._width is not None and len(row) != self._width:
            raise ValueError(f"Length of row ({len(row)}) doesn't match the width of {self.__class__.__name__} ({self._width})")
        row_id = len(self._rows)
        self._rows.append(tuple(row))
        return row_id

    def get_sprites(self, g):
        if not self._rows:
            raise RuntimeError(f'Trying to get sprites for empty {self.__class__.__name__}')
        res = []
        res.append(grf.Action1(
            feature=self.feature,
            set_count=len(self._rows),
            sprite_count=self._width,
        ))
        for r in self._rows:
            res.extend(r)
        return res


class VehicleSpriteTable(SpriteTable):
    def __init__(self, feature):
        assert feature in grf.VEHICLE_FEATURES
        super().__init__(feature)
        self._width = 8

    def get_layout(self, row_id):
        return grf.GenericSpriteLayout(ent1=(row_id,), ent2=(row_id,))

    def add_purchase_graphics(self, sprite, second_head=None):
        return self.add_row((
            grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            second_head or grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            sprite,
            grf.EMPTY_SPRITE
        ))


def fake_vehicle_info(props):
    return '{}'.join('{BLACK}' + k + ': {GOLD}' + v for k, v in props.items())


class Vehicle(grf.SpriteGenerator):

    class ExtraFlags:
        DISABLE_NEW_VEHICLE_MESSAGE = 0x01
        DISABLE_EXCLUSIVE_PREVIEW = 0x02
        SYNC_VARIANT_EXCLUSIVE_PREVIEW = 0x04
        SYNC_VARIANT_RELIABILITY = 0x08

    def __init__(self, feature, *, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None):
        self._validate_liveries(liveries)

        self.feature = feature
        self.liveries = liveries
        self.callbacks = make_callback_manager(self.feature, callbacks)
        self.purchase_sprite = purchase_sprite
        self.additional_text = additional_text

    def _validate_liveries(self, liveries):
        for l in liveries or []:
            if 'name' not in l:
                raise ValueError(f'{self.__class__.__name__} livery is missing the name')
            sprites = l.get('sprites')
            if sprites is None:
                raise ValueError(f'{self.__class__.__name__} livery {l["name"]} is missing sprites')
            if len(sprites) != 8:
                raise ValueError(f'{self.__class__.__name__} livery expects 8 sprites, found {len(sprites)}')

    def _gen_name_sprites(self, g, vehicle_id):
        if isinstance(self.name, grf.StringRef):
            return self.name.get_actions(self.feature, vehicle_id)
        else:
            return g.strings.add(self.name).get_actions(self.feature, vehicle_id)

    def _gen_purchase_sprites(self):
        res = []

        if self.purchase_sprite:
            # Make sure purchase layouts go before main action 1
            # TODO make Action1 referenceable and move to _set_callbacks
            res.append(grf.SpriteSet(feature=self.feature, count=1))
            res.append(self.purchase_sprite)
            res.append(layout := grf.GenericSpriteLayout(ent1=(0,), ent2=(0,)))
            self.callbacks.graphics.purchase = layout
        return res

    def _set_callbacks(self, g):
        if self.additional_text:
            self.callbacks.additional_text = g.strings.add(self.additional_text).get_global_id()
        return []


class VariantGroup:
    def __init__(self, name, *items):
        self.name = name
        if len(items) < 2:
            raise ValueError('VariantGroup needs at least 2 vehicles')
        if isinstance(items[0], VariantGroup):
            raise ValueError('First element of a VariantGroup can''t be a VariantGroup')
        self.items = items

    @property
    def first(self):
        x = self.items[0]
        if isinstance(x, VariantGroup):
            return x.first
        return x


class SetPurchaseOrder(grf.SpriteGenerator):
    def __init__(self, *order):
        super().__init__()
        self.flat_order = []
        self._variant_group = {}
        self._variant_callbacks = defaultdict(dict)
        self._variant_callbacks_set = False
        self._feature = None

        def process(order, group, level, callbacks):
            for e in order:
                if isinstance(e, VariantGroup):
                    x = e.first
                    if e.name is not None:
                        cb = {
                            **callbacks,
                            level: e.name,
                        }
                    else:
                        cb = callbacks

                    process(e.items, x, level + 1, cb)
                    if group is not None:
                        self._variant_group[id(x)] = group
                    else:
                        self._variant_group.pop(id(x))
                else:
                    self.flat_order.append(e)
                    if group is not None:
                        self._variant_group[id(e)] = group
                    if callbacks:
                        self._variant_callbacks[id(e)] = (e, callbacks)
                    if self._feature is None:
                        self._feature = e.feature

        process(order, None, 0, {})
        if self._feature is None:
            raise ValueError('Could not detect feature.')

    def set_variant_callbacks(self, g):
        for _, (x, names) in self._variant_callbacks.items():
            c = x.callbacks.name = grf.Switch(
                code='extra_callback_info1',
                ranges = {
                    0x20 | (level << 8): g.strings.add(name).get_global_id()
                    for level, name in names.items()
                },
                default=0x400,
            )
        self._variant_callbacks_set = True
        return self

    def get_sprites(self, g):
        if not self._variant_callbacks_set:
            raise RuntimeError(
                'SetPurchaseOrder.set_variant_callbacks needs to be called before the'
                ' sprite generation when using custom variant group names.')
        prev = None
        res = []
        for v in reversed(self.flat_order):
            variant_props = {}
            sort_props = {}
            if prev is not None:
                sort_props = {'sort_purchase_list': g.resolve_id(self._feature, prev.id)}

            vg = self._variant_group.get(id(v))
            if vg is not None:
                variant_props = {'variant_group': g.resolve_id(self._feature, vg.id)}

            if sort_props or variant_props:
                res.append(grf.Define(
                    feature=self._feature,
                    id=g.resolve_id(self._feature, v.id),
                    props={**sort_props, **variant_props}
                ))
            prev = v
        return res


class EngineOverride(grf.SpriteGenerator):
    def __init__(self, target, *, source=None):
        self.target = target
        self.source = source

    def py(self, context):
        return f'EngineOverride(target={self.target!r}, source={self.source!r})'

    def get_sprites(self, g):
        source = self.source if self.source is not None else g.grfid
        return [
            grf.Define(
                feature=grf.GLOBAL_VAR,
                id=0,
                props={
                    'grfid_overrides': (source, self.target),
                }
            )
        ]
