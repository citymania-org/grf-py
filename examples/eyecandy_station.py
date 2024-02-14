import grf


class Station(grf.SpriteGenerator):
    NO_TILES, ALL_TILES = 0, 255

    def __init__(self, name, class_label, class_name, sprite, extent, callbacks=None, **props):
        super().__init__()
        self.callbacks = grf.make_callback_manager(grf.STATION, callbacks)
        self.id = None
        self.name = name
        self.sprite = sprite
        self.extent = extent
        self.class_label = grf.to_bytes(class_label)
        self.class_name = class_name
        self._props = props

    def get_sprites(self, g):
        res = []

        layout_x = grf.SpriteLayout([
            grf.GroundSprite(
                sprite=grf.SpriteRef(id=1420, pal=0, is_global=True, use_recolour=False, always_transparent=False, no_transparent=False),
                flags=0,
            ),
            grf.ParentSprite(
                sprite=grf.SpriteRef(id=0x42d, pal=0, is_global=False, use_recolour=True, always_transparent=False, no_transparent=False),
                extent=self.extent,
                offset=(0, 0, 0),
                flags=0,
            )
        ])

        res.append(definition := grf.Define(
            feature=grf.STATION,
            id=self.id,
            props={
                'class_label': self.class_label,
                'advanced_layout': grf.SpriteLayoutList([layout_x, layout_x]),
                **self._props
            }
        ))

        res.append(grf.DefineStrings(
            feature=grf.STATION,
            offset=0xc400 + self.id,
            is_generic_offset=True,
            strings=[self.class_name.encode('utf-8')]
        ))
        res.append(grf.DefineStrings(
            feature=grf.STATION,
            offset=0xc500 + self.id,
            is_generic_offset=True,
            strings=[self.name.encode('utf-8')]
        ))

        res.append(grf.Action1(
            feature=grf.STATION,
            set_count=1,
            sprite_count=1,
        ))
        res.append(self.sprite)
        layout = grf.GenericSpriteLayout(
            ent1=(),
            ent2=(0,),
        )
        res.append(grf.Map(
            definition,
            {},
            layout,
        ))
        return res


class EyecandyStation(Station):
    def __init__(self, name, class_label, class_name, sprite, extent, callbacks=None, **props):
        super().__init__(
            name, class_label, class_name, sprite, extent, callbacks=callbacks,
            draw_pylon_tiles=Station.NO_TILES,
            hide_wire_tiles=Station.ALL_TILES,
            non_traversable_tiles=Station.ALL_TILES,
            **props
        )


# TODO this is very ugly way to assign ids, needs to be switched to auto-id
def init_stations(g):
    next_station_id = 0
    for s in g.generators:
        if isinstance(s, Station):
            s.id = next_station_id
            next_station_id += 1


g = grf.NewGRF(
    grfid=b'GPE\x08',
    name='grf-py eyecandy station example',
    description='grf-py eyecandy station example',
)

house_png = grf.ImageFile('sprites/vox_house1460.png')

g.add(EyecandyStation(
    name='Test station',
    class_name='Test station class',
    class_label=b'TSCL',
    sprite=grf.FileSprite(house_png, 5, 5, 48, 60, xofs=-25, yofs=-33),
    extent=(16, 16, 34),
))

init_stations(g)
g.write('grfpy_example_eyecandy_station.grf')
