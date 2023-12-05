from typing import Optional, Union

from typeguard import typechecked


import grf
from .callback import make_callback_manager


ID = Union[str, int]
Size = tuple[int, int]
String = Union[str, grf.StringRef]


class Industry(grf.SpriteGenerator):

    class Building:
        @typechecked
        def __init__(
            self,
            *,
            size: tuple[int, int],
            sprite: grf.Sprite,
        ):
            self.size = size
            self.sprite = sprite

        def get_sprites(self):
            w, h = self.size

            tiles = {}
            sprites = []

            def add_strip(lx, ly, x, w, xofs, yofs):
                delta = 1
                MAX_HEIGHT = 231
                ybottom = self.sprite.h
                while delta > 0 and ybottom > 0:
                    ytop = max(ybottom - MAX_HEIGHT, 0)
                    delta = min(lx - 1, ly - 1, 7)
                    if delta == 0:
                        ytop = 0
                    tiles[lx, ly] = len(sprites)
                    sprites.append(
                        grf.FileSprite(
                            self.sprite.file,
                            self.sprite.x + x,
                            self.sprite.y + ytop,
                            w,
                            ybottom - ytop,
                            bpp=self.sprite.bpp,
                            xofs=xofs,
                            yofs=self.sprite.yofs + yofs + ytop,
                        )
                    )
                    ybottom = ytop
                    lx -= delta
                    ly -= delta
                    yofs += delta * 32

            for i in range(1, h):
                xb = -self.sprite.xofs - 32 * (h - i + 1) + 1
                add_strip(w, i, xb, 32, xofs=-31, yofs=(h - i) * 16)

            add_strip(w, h, -self.sprite.xofs - 31, 64, xofs=-31, yofs=0)

            for i in range(1, w):
                xb = -self.sprite.xofs + i * 32 + 1
                add_strip(w - i, h, xb, 32, xofs=1, yofs=i * 16)

            return tiles, sprites

    @typechecked
    def __init__(
        self,
        *,
        id: ID,
        name: String,
        substitute_type: int,
        layouts: list[list[tuple[int, int, 'Industry.Building']]],
        z_extent: int,
        ground_sprite_id: int,
        production_types=[],
        acceptance_types=[],
        callbacks=None,
        **props,
    ):
        super().__init__()
        self.id = id
        self.name = name
        self.substitute_type = substitute_type
        self.layouts = layouts
        self.z_extent = z_extent
        self.ground_sprite_id = ground_sprite_id
        self._props = props
        self.callbacks = make_callback_manager(grf.INDUSTRY, callbacks)
        self.acceptance_types = acceptance_types
        self.production_types = production_types

    @typechecked
    def get_sprites(self, g: grf.NewGRF):
        res = []
        if isinstance(self.name, grf.StringRef):
            name_id = self.name
        else:
            name_id = g.strings.add(self.name).get_persistent_id()
        industry_id = g.resolve_id(grf.INDUSTRY, self.id)
        tile_id = g.resolve_id(grf.INDUSTRY_TILE, f'__industry_{industry_id}_main_tile')
        res.append(tile_definition := grf.Define(
            feature=grf.INDUSTRY_TILE,
            id=tile_id,
            props={
                'substitute_type': 0,
                'land_shape_flags': 0,
            }
        ))

        sprites = [grf.EMPTY_SPRITE]
        building_cache = {}
        layout_switches = []
        layout_tiles = []
        for i, l in enumerate(self.layouts):
            tiles = {}
            occupied_tiles = set()
            for x0, y0, b in l:
                bdata = building_cache.get(b)
                if bdata is None:
                    tl, sl = b.get_sprites()
                    sofs = len(sprites)
                    sprites.extend(sl)
                    building_cache[b] = tl, sofs
                else:
                    tl, sofs = bdata

                for x in range(b.size[0]):
                    for y in range(b.size[0]):
                        key = x + x0, y + y0
                        if key in occupied_tiles:
                            raise RuntimeError(f'Overlapping buildings in industry {self.id} layout {i}')
                        occupied_tiles.add(key)

                for (x, y), n in tl.items():
                    key = x + x0 - 1, y + y0 - 1
                    tiles[key] = n + sofs
            layout_tiles.append(occupied_tiles)
            layout_switches.append(grf.Switch(
                feature=grf.INDUSTRY_TILE,
                ranges={(y << 8) | x : n for (x, y), n in tiles.items()},
                default=0,
                code='relative_pos',
            ))

        res.append(grf.Action1(
            feature=grf.INDUSTRY_TILE,
            set_count=1,
            sprite_count=len(sprites),
        ))
        res.extend(sprites)

        get_sprite_num = grf.Switch(
            feature=grf.INDUSTRY_TILE,
            ref_id=0,  # TODO fix collisions
            ranges=dict(enumerate(layout_switches)),
            default=layout_switches[0],
            code='layout_num',
            related_scope=True,
        )

        # TODO needs different layout for each tile for extent/offset (but cache per building)
        layout = grf.AdvancedSpriteLayout(
            feature=grf.INDUSTRY_TILE,
            ground={
                'sprite': grf.SpriteRef(self.ground_sprite_id, is_global=True),
            },
            buildings=[{
                'sprite': grf.SpriteRef(0, is_global=False),
                'add': grf.Temp(0),
                'extent': (16, 16, self.z_extent),
            }],
        )

        layout_switch = grf.Switch(
            code=f'''
                TEMP[0] = get_sprite_num()
            ''',
            ranges={0: layout},
            default=layout,
            subroutines={
                'get_sprite_num': get_sprite_num,
            }
        )

        res.append(grf.Map(
            tile_definition,
            maps={255: layout_switch},
            default=layout_switch,
        ))

        props = self._props.copy()
        self.callbacks.set_flag_props(props)

        res.append(definition := grf.Define(
            feature=grf.INDUSTRY,
            id=industry_id,
            props={
                'substitute_type': self.substitute_type,
                'name': name_id,
                'layouts': [
                    grf.IndustryLayout([
                        grf.IndustryLayout.NewTile(
                            xofs=x,
                            yofs=y,
                            id=tile_id,
                        )
                        for x, y in lt
                    ])
                    for lt in layout_tiles
                ],
                'acceptance_types': [g.get_cargo_id(x) for x in self.acceptance_types],
                'production_types': [g.get_cargo_id(x) for x in self.production_types],
                **props,
            },
        ))

        res.extend(self.callbacks.make_map_action(definition))

        return res