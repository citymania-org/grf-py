# pip install git+https://github.com/citymania-org/grf-py.git@4c88bc7d909e317187561392b38511c98989bc55#egg=grf


from datetime import date

from PIL import Image, ImageDraw

import grf
import math

# Teach grf-py new callback that PR#11489 adds
grf.CALLBACKS[grf.TRAIN][0x162] = {
    'name': 'build_flipped_probability',
    'flag': 0,
    'class': grf.DefaultCallback,
}


g = grf.NewGRF(
    grfid=b'GPE\x06',
    name='Flipping probability CB test (#11489)',
    description='Test NewGRF for PR #11489. Chance to build flipped = year - 1900',
)
Train = g.bind(grf.Train)


class TrainSprite(grf.ImageSprite):
    def __init__(self, direction):
        w, h = 17, 17
        img = Image.new('P', (w, h))
        img.putpalette(grf.PALETTE)
        imd = ImageDraw.Draw(img)
        imd.rectangle((0, 0, w - 1, h - 1), fill=0x43, outline=1)
        cx, cy = w // 2, h // 2
        line = lambda r1, a1, r2, a2: imd.line(
            (
                (int(cx + r1 * math.sin(-a1)), int(cy + r1 * math.cos(-a1))),
                (int(cx + r2 * math.sin(-a2)), int(cy + r2 * math.cos(-a2)))
            ),
            width=1,
            fill=1
        )
        a = direction * math.pi / 4
        line(-6, a, 6, a)
        line(-6, a, 5, a + math.pi / 3)
        line(-6, a, 5, a - math.pi / 3)
        super().__init__(img, xofs=-8, yofs=-8)


t = Train(
    id='random-reverse-1',
    name='Random Reverse',
    liveries=[{
        'name': 'Default',
        'sprites': [TrainSprite(i) for i in range(8)],
    }],
    engine_class=Train.EngineClass.DIESEL,
    max_speed=Train.kmhish(104),
    power=0,
    introduction_date=date(1800, 1, 1),
    weight=20,
    tractive_effort_coefficient=79,
    vehicle_life=8,
    model_life=144,
    climates_available=grf.ALL_CLIMATES,
    running_cost_factor=222,
    cargo_capacity=90,
    default_cargo_type=0,
    cost_factor=1,
    refittable_cargo_types=1,
)

t.callbacks.build_flipped_probability = grf.Eval(
    'current_year - 1900',
    default=0,
)

grf.main(g, 'pr11489.grf')
