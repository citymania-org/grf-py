import grf

g = grf.NewGRF(
    grfid=b'GPE\x02',
    name='grf-py VOX house example',
    description='grf-py VOX house example',
)

vox = grf.VoxFile("sprites/house1460.vox")
g.add(grf.ReplaceOldSprites([(1460, 1)]))
g.add(
    vox.make_house_sprite(zoom=grf.ZOOM_4X, xofs=-25, yofs=-33),
    vox.make_house_sprite(zoom=grf.ZOOM_2X, xofs=-25 * 2, yofs=-33 * 2),
    vox.make_house_sprite(zoom=grf.ZOOM_NORMAL, xofs=-25 * 4, yofs=-33 * 4),
)
g.write('vox_house.grf')

