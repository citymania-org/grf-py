from datetime import date

import grf

g = grf.NewGRF(
    grfid=b'GPE\x05',
    name='grf-py simple industry example',
    description='grf-py simple industry example',
)

Industry = g.bind(grf.Industry)

img = grf.ImageFile("sprites/gift_box.png")
sprite = grf.FileSprite(img, 1, 1, 256, 282, xofs=-127, yofs=-251)

Industry(
    id=0,
    substitute_type=0,
    name='A Simple Industry',
    layouts=[
        [(0, 0, Industry.Building(size=(4, 4), sprite=sprite))],
    ],
    z_extent=154,
    ground_sprite_id=1420,
)

grf.main(g, 'grfpy_example_industry_simple.grf')
