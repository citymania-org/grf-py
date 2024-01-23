# pip install git+https://github.com/citymania-org/grf-py.git@ae4aeab54638cc206d3c969caae0e473a8636651#egg=grf

import struct

import grf

THIS_FILE = grf.PythonFile(__file__)


class SVGFile(grf.ResourceFile):
    def get_data(self):
        return open(self.path, 'rb').read()


class SVGSprite(grf.Resource):
    def __init__(self, file, w, h, xofs, yofs):
        assert(isinstance(file, SVGFile))

        super().__init__()
        self.file = file
        self.w = w
        self.h = h
        self.xofs = xofs
        self.yofs = yofs

    def get_real_data(self, context):
        data = self.file.get_data()
        return struct.pack(
            '<BBHHhhI',
            0xFE,  # type = 0xFE
            0,  # zoom = ignored
            self.h,
            self.w,
            self.xofs,
            self.yofs,
            len(data) + 1,
        ) + data + b'\0'

    def get_fingerprint(self):
        return dict(
            **{'class': self.__class__.__name__},
            w=self.w,
            h=self.h,
            xofs=self.xofs,
            yofs=self.yofs,
        )

    def get_resource_files(self):
        return (THIS_FILE, self.file)


g = grf.NewGRF(
    grfid=b'GPE\xFF',
    name='Test SVG sprites',
    description='Test SVG sprites',
)

g.add(grf.ReplaceOldSprites(((726, 1),)))
# g.add(grf.AlternativeSprites(
#     SVGSprite(SVGFile('pause.svg'), 1600, 1600, 0, 0),
#     grf.FileSprite(grf.ImageFile('pause.png'), 0, 0, 16, 16),
# ))
g.add(SVGSprite(SVGFile('pause.svg'), 16, 16, 0, 0))
# g.add(grf.FileSprite(grf.ImageFile('pause.png'), 0, 0, 16, 16))

grf.main(g, 'test_svg_sprites.grf')
