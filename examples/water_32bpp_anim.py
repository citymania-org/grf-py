import grf

WATER_SPRITE_ID = 4061

g = grf.NewGRF(
    grfid=b'GPE\x01',
    name='grf-py Palette-animated 32bpp water example',
    description='grf-py Palette-animated 32bpp water example',
)

water_png = grf.ImageFile("sprites/water_noise.png", colourkey=(0, 0, 255))
water_mask_png = grf.ImageFile("sprites/ogfx_water.png")
g.add(grf.ReplaceOldSprites([(WATER_SPRITE_ID, 1)]))
g.add(grf.WithMask(
    grf.FileSprite(water_png, 1, 1, 64, 31, xofs=-31, bpp=24),
    grf.FileSprite(water_mask_png, 1, 1, 64, 31),
))
g.write('water_32bpp_anim.grf')

