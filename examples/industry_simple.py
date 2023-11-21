from datetime import date
from typing import Optional, Union

from typeguard import typechecked

import grf

g = grf.NewGRF(
    grfid=b'GPE\x05',
    name='grf-py simple industry example',
    description='grf-py simple industry example',
)

ID = Union[str, int]
Size = tuple[int, int]
String = Union[str, grf.StringRef]

class Industry(grf.SpriteGenerator):
    @typechecked
    def __init__(
        self,
        *,
        id: ID,
        name: String,
        size: Size,
        substitute_type: int,
        sprite: Optional[grf.Sprite] = None,
        **props,
    ):
        super().__init__()
        self.id = id
        self.name = name
        self.size = size
        self.substitute_type = substitute_type
        self.sprite = sprite
        self._props = props

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
            id=0,
            props={
                'substitute_type': 0,
                'land_shape_flags': 0,
            }
        ))

        w, h = self.size

        buildings = {}

        def add_strip(lx, ly, x, w, xofs, yofs):
            delta = 1
            MAX_HEIGHT = 231
            ybottom = self.sprite.h
            while delta > 0 and ybottom > 0:
                ytop = max(ybottom - MAX_HEIGHT, 0)
                delta = min(lx - 1, ly - 1, 7)
                if delta == 0:
                    ytop = 0
                buildings[lx, ly] = len(sprites)
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


        sprites = [grf.EMPTY_SPRITE]
        for i in range(1, h):
            xb = -self.sprite.xofs - 32 * (h - i + 1) + 1
            add_strip(w, i, xb, 32, xofs=-31, yofs=(h - i) * 16)

        add_strip(w, h, -self.sprite.xofs - 31, 64, xofs=-31, yofs=0)

        for i in range(1, w):
            xb = -self.sprite.xofs + i * 32 + 1
            add_strip(w - i, h, xb, 32, xofs=1, yofs=i * 16)

        res.append(grf.Action1(
            feature=grf.INDUSTRY_TILE,
            set_count=1,
            sprite_count=len(sprites),
        ))
        res.extend(sprites)

        get_sprite_num = grf.Switch(
            feature=grf.INDUSTRY_TILE,
            ref_id=0, # TODO fix collisions
            ranges={((y - 1) << 8) | (x - 1) : n for (x, y), n in buildings.items()},
            default=0,
            code='relative_pos',
        )
        res.append(get_sprite_num)

        layout = grf.AdvancedSpriteLayout(
            feature=grf.INDUSTRY_TILE,
            ground={
                'sprite': grf.SpriteRef(4550, is_global=True),
            },
            buildings=[{
                'sprite': grf.SpriteRef(0, is_global=False),
                'add': grf.Temp(0),
                'extent': (16, 16, 154),
            }],
        )

        layout_switch = grf.Switch(
            code=f'''
                TEMP[0] = call({get_sprite_num})
            ''',
            ranges={0: layout},
            default=layout,
        )

        res.append(grf.Map(
            tile_definition,
            maps={255: layout_switch},
            default=layout_switch,
        ))

        res.append(grf.Define(
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
                            id=0,
                        )
                        for x in range(w)
                        for y in range(h)
                    ])
                ],
                **self._props,
            },
        ))
        return res


# Industry = g.bind(grf.Industry)
Industry = g.bind(Industry)

def tmpl_rv(x, y, func):
    return [
        func(      x, y, 10, 28, xofs= -4, yofs=-15),
        func( x + 20, y, 26, 28, xofs=-18, yofs=-14),
        func( x + 50, y, 36, 28, xofs=-18, yofs=-17),
        func( x + 90, y, 26, 28, xofs=-10, yofs=-15),
        func(x + 120, y, 10, 28, xofs= -4, yofs=-15),
        func(x + 140, y, 26, 28, xofs=-16, yofs=-16),
        func(x + 170, y, 36, 28, xofs=-18, yofs=-20),
        func(x + 210, y, 26, 28, xofs= -6, yofs=-15),
    ]

img = grf.ImageFile("sprites/gift_box.png")
sprite = grf.FileSprite(img, 1, 1, 256, 282, xofs=-127, yofs=-251)

Industry(
    id=0,
    substitute_type=0,
    name='A Simple Industry',
    size=(4, 4),
    sprite=sprite,
)

g.write('grfpy_example_industry_simple.grf')
