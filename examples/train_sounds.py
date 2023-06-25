from datetime import date

import grf

g = grf.NewGRF(
    grfid=b'GPE\x03',
    name='grf-py example of a train with sounds',
    description='grf-py example of a train with sounds',
)
Train = g.bind(grf.Train)

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

rv_png = grf.ImageFile('sprites/32bpp_rv.png', colourkey=(0, 0, 255))
sprites = tmpl_rv(0, 20, lambda *args, **kw: grf.FileSprite(rv_png, *args, **kw, bpp=24))

diesel_effects = {
    grf.SoundEvent.STOPPED: grf.RAWSound('sounds/modern_diesel_idle.wav'),
    grf.SoundEvent.VISUAL_EFFECT: grf.RAWSound('sounds/modern_diesel_run.wav'),
    grf.SoundEvent.RUNNING_16: grf.RAWSound('sounds/modern_diesel_coast.wav'),
    grf.SoundEvent.START: grf.RAWSound('sounds/horn_4.wav'),
    grf.SoundEvent.BREAKDOWN: grf.DefaultSound.BREAKDOWN_TRAIN_SHIP,
    grf.SoundEvent.TUNNEL: grf.RAWSound('sounds/horn_4.wav'),  # sounds are cached by filename so horn_4 will only be added once
}

Train(
    id=300,
    name='Example train with sounds',
    liveries=[{
        'name': 'Default',
        'sprites': sprites,
    }],
    sound_effects=diesel_effects,
    engine_class=Train.EngineClass.DIESEL,
    max_speed=Train.kmhish(104),
    power=255,
    introduction_date=date(1900, 1, 1),
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

g.write('grfpy_train_sound_example.grf')
