from datetime import date

import grf

g = grf.NewGRF(
    grfid=b'GPE\x00',
    name='grf-py 2cc 32bpp road vehicle example',
    description='grf-py 2cc 32bpp road vehicle example',
)
g.strings = grf.StringManager()
RoadVehicle = g.bind(grf.RoadVehicle)

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

rv_png = grf.ImageFile("sprites/32bpp_rv.png", colourkey=(0, 0, 255))
rv_mask_png = grf.ImageFile("sprites/32bpp_rv_mask.png")

RoadVehicle(
    id=5,
    name='2cc 32bpp road vehicle example',
    liveries=[{
        'name': 'Default',
        'sprites': tmpl_rv(0, 20, lambda *args, **kw: grf.FileSprite(rv_png, *args, **kw, bpp=24,
                                                                     mask=(rv_mask_png, 0, 0))),
    }],
    introduction_date=date(1900, 1, 1),
    max_speed=RoadVehicle.kmhishph(200),
    vehicle_life=8,
    model_life=144,
    climates_available=grf.ALL_CLIMATES,
    running_cost_factor=222,
    cargo_capacity=90,
    default_cargo_type=0,
    cost_factor=246,
    misc_flags=RoadVehicle.Flags.USE_2CC,
)

g.add(g.strings)
g.write('grfpy_2cc_32bpp_rv_example.grf')
