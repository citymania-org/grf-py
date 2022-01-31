import grf

WATER_SPRITE_ID = 4061

g = grf.NewGRF(
    b'GPE\x01',
    8,
    'grf-py Palette-animated 32bpp water example',
    'grf-py Palette-animated 32bpp water example',
)

water_png = grf.ImageFile("sprites/water_noise.png", colourkey=(0, 0, 255))
water_mask_png = grf.ImageFile("sprites/ogfx_water.png")
g.add(grf.ReplaceOldSprites([(WATER_SPRITE_ID, 1)]))
g.add(grf.FileSprite(water_png, 1, 1, 64, 31, xofs=-31, bpp=24, mask=(water_mask_png, 0, 0)))
g.write('water_32bpp_anim.grf')

