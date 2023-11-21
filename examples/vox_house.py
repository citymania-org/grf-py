import grf
from PIL import Image

g = grf.NewGRF(
    grfid=b'GPE\x04',
    name='grf-py VOX house example',
    description='grf-py VOX house example',
)

def debug_sprites(sprites, scale):
    rx, ry = 0, 0

    for s, sscale in sprites:
        rx += s.w * sscale
        ry = max(ry, s.h * sscale)
    im = Image.new('RGBA', (rx + 10 * len(sprites) - 10, ry))
    x = 0
    for s, sscale in sprites:
        simg = s.get_image()[0]
        simg = simg.resize((simg.size[0] * sscale, simg.size[1] * sscale), Image.NEAREST)
        im.paste(simg, (x, 0))
        x += s.w * sscale + 10
    im = im.resize((im.size[0] * scale, im.size[1] * scale), Image.NEAREST)
    im.show()

vox = grf.VoxFile("sprites/house1460.vox")
g.add(grf.ReplaceOldSprites([(1460, 1)]))
g.add(
    vox.make_house_sprite(zoom=grf.ZOOM_4X, xofs=-25, yofs=-33),
    vox.make_house_sprite(zoom=grf.ZOOM_2X, xofs=-25 * 2, yofs=-33 * 2),
    vox.make_house_sprite(zoom=grf.ZOOM_NORMAL, xofs=-25 * 4, yofs=-33 * 4),
)

debug_sprites((
    (vox.make_house_sprite(zoom=grf.ZOOM_4X, xofs=-25, yofs=-33), 4),
    (vox.make_house_sprite(zoom=grf.ZOOM_2X, xofs=-25 * 2, yofs=-33 * 2), 2),
    (vox.make_house_sprite(zoom=grf.ZOOM_NORMAL, xofs=-25 * 4, yofs=-33 * 4), 1)),
    1)

g.write('vox_house.grf')

