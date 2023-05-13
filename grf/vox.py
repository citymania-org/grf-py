import struct
import numpy as np
from PIL import Image, ImageDraw

from .common import to_spectra, ZOOM_4X, ZOOM_2X, ZOOM_NORMAL, PALETTE
from .sprites import ImageSprite, convert_image, PaletteRemap, find_best_color


VOX_SIDE_RIGHT, VOX_SIDE_BOTTOM = 0x100, 0x200
VOX_SIDE_SHIFT, VOX_SIDE_MASK = 8, VOX_SIDE_RIGHT | VOX_SIDE_BOTTOM
VOX_SIDE_ZL, VOX_SIDE_ZR, VOX_SIDE_X, VOX_SIDE_Y = 0, VOX_SIDE_RIGHT, VOX_SIDE_BOTTOM, VOX_SIDE_RIGHT | VOX_SIDE_BOTTOM
VOX_SIDE_Z, VOX_SIDE_XY = 0, 0x100


class VoxReader:
    def __init__(self, path):
        self.path = path
        self.size = None
        self.voxels = []
        self.palette = None

    def _read_size(self, f):
        s = struct.unpack('<III', f.read(12))
        self.size = [s[1], s[0], s[2]]

    def _read_xyzi(self, f):
        n = struct.unpack('<I', f.read(4))[0]
        self.voxels = np.fromfile(f, count=n * 4, dtype=np.uint8)
        self.voxels.shape = (n, 4)

        # x, y, z, i = sx - y, x, z, i
        # self.voxels[:, [0, 1]] = self.voxels[:, [1, 0]]
        self.voxels[:, 1] = self.size[1] - self.voxels[:, 1]
        self.voxels = np.array((self.size[0], 0, 0, 0)) + self.voxels
        assert np.all(self.voxels[:,3] > 0) # index 0 seems to be unused

    def _read_rgba(self, f):
        vox_palette = np.fromfile(f, count=255 * 4, dtype=np.uint8)
        f.read(4) # last colour is unused and pal starts with index 1
        vox_palette = np.concatenate(([0, 0, 0, 0], vox_palette))
        vox_palette.shape = (256, 4)
        self.vox_palette = vox_palette
        self.palette = [to_spectra(r, g, b) for r, g, b, _ in vox_palette]
        self.ttd_palette = [find_best_color(c) for c in self.palette]

    def read(self):
        with open(self.path, 'rb') as f:
            f.read(4)  # VOX
            f.read(4)  # Version
            while True:
                chunk_id = f.read(4)
                if not chunk_id:
                    break
                n, m = struct.unpack('<II', f.read(8))
                pos = f.tell()
                if chunk_id == b'XYZI':
                    self._read_xyzi(f)
                if chunk_id == b'SIZE':
                    self._read_size(f)
                elif chunk_id == b'RGBA':
                    self._read_rgba(f)
                f.seek(pos + n, 0)


class VoxFile:
    def __init__(self, path, *,
                 x_colour_func=lambda c: c.darken(20),
                 y_colour_func=None,
                 z_colour_func=lambda c: c.brighten(10)):
        if isinstance(path, VoxReader):
            self.reader = path
            self.path = None
        else:
            self.path = path
            self.reader = VoxReader(path)

        self.x_colour_func = x_colour_func
        self.y_colour_func = y_colour_func
        self.z_colour_func = z_colour_func

        self.data = None

    def get_image(self):
        pass

    def _load(self):
        self.reader.read()
        cam_norm = np.array((6**.5 / 4., 6**.5 / 4., .5, 0))
        xx = self.reader.voxels @ np.array((1, -1, 0, 0)).T
        yy = self.reader.voxels @ np.array((1, 1, -2, 0)).T
        zz = self.reader.voxels @ cam_norm
        cc = np.take(self.reader.ttd_palette, self.reader.voxels[:, 3])
        xmin, xmax = np.amin(xx), np.amax(xx)
        ymin, ymax = np.amin(yy), np.amax(yy)
        self.w = xmax - xmin + 2
        self.h = ymax - ymin + 3
        zbuf = np.full((self.h, self.w), -1e100)
        data = np.zeros((self.h, self.w), dtype=np.uint16)

        def set_pixel(x, y, z, c):
            if zbuf[y, x] >= z:
                return
            zbuf[y, x] = z
            data[y, x] = c

        for x, y, z, c in zip(xx, yy, zz, cc):
            if c == 1: continue
            x -= xmin
            y -= ymin
            set_pixel(x, y, z, c | VOX_SIDE_ZL)
            set_pixel(x + 1, y, z, c | VOX_SIDE_ZR)
            set_pixel(x, y + 1, z, c | VOX_SIDE_X)
            set_pixel(x + 1, y + 1, z, c | VOX_SIDE_Y)
            set_pixel(x, y + 2, z, c | VOX_SIDE_X)
            set_pixel(x + 1, y + 2, z, c | VOX_SIDE_Y)

        self.data = data

        def make_remap(func):
            if func is None:
                return np.arange(256)
            return PaletteRemap.from_function(func).remap

        REMAP_X = make_remap(self.x_colour_func)
        REMAP_Y = make_remap(self.y_colour_func)
        REMAP_Z = make_remap(self.z_colour_func)
        self.palette = np.concatenate((REMAP_Z, REMAP_Z, REMAP_X, REMAP_Y)).astype(np.uint8)

    def make_house_sprite(self, zoom, **kw):
        if self.data is None:
            self._load()

        if zoom == ZOOM_4X:
            data = self.palette[self.data[::2, :]]
        elif zoom == ZOOM_2X:
            data = self.palette[np.repeat(self.data, repeats=2, axis=1)]
        elif zoom == ZOOM_NORMAL:
            data = np.zeros((self.data.shape[0] * 2 + 1, self.data.shape[1] * 4), dtype=np.uint8)
            SIDE_PIXELS = (
                (                (2, 0), (3, 0),  # Z left
                 (0, 1), (1, 1), (2, 1), (3, 1),
                                 (2, 2), (3, 2)),
                ((0, 0), (1, 0), # Z right
                 (0, 1), (1, 1), (2, 1), (3, 1),
                 (0, 2), (1, 2)),
                ((0, 0), (1, 0),  # X (default)
                 (0, 1), (1, 1), (2, 1), (3, 1),
                                 (2, 2), (3, 2)),
                (                (2, 0), (3, 0),  # Y (default)
                 (0, 1), (1, 1), (2, 1), (3, 1),
                 (0, 2), (1, 2)),
            )
            # Another pattern
            # SIDE_PIXELS = (
            #     (                        (3, 0),  # Z left
            #              (1, 1), (2, 1), (3, 1),
            #              (1, 2), (2, 2), (3, 2),
            #                              (3, 3)),
            #     ((0, 0),  # Z right
            #      (0, 1), (1, 1), (2, 1),
            #      (0, 2), (1, 2), (2, 2),
            #      (0, 3)),
            #     ((0, 0),  # X
            #      (0, 1), (1, 1), (2, 1),
            #              (1, 2), (2, 2), (3, 2),
            #                              (3, 3)),
            #     (                        (3, 0),  # Y
            #              (1, 1), (2, 1), (3, 1),
            #      (0, 2), (1, 2), (2, 2),
            #      (0, 3)),
            # )
            for y in range(self.h):
                for x in range(self.w):
                    yy = 2 * y
                    xx = 4 * x
                    v = self.data[y, x]
                    colour = self.palette[v]
                    side = v >> VOX_SIDE_SHIFT

                    if v & VOX_SIDE_BOTTOM > 0:
                        # If there is same Z side below use triangle pattern
                        if (y + 1 < self.h and self.data[y + 1, x] and
                            self.data[y + 1, x] & VOX_SIDE_BOTTOM == 0 and
                            self.data[y + 1, x] & VOX_SIDE_RIGHT == v & VOX_SIDE_RIGHT):
                            # SIDE_X -> SIDE_ZR   SIDE_Y -> SIDE_ZL
                            side ^= ((VOX_SIDE_BOTTOM | VOX_SIDE_RIGHT) >> VOX_SIDE_SHIFT)

                        # If X is below Y or vice versa use triangle pattern
                        if (y > 0 and self.data[y - 1, x] and
                            self.data[y - 1, x] & VOX_SIDE_BOTTOM > 0 and
                            self.data[y - 1, x] & VOX_SIDE_RIGHT != v & VOX_SIDE_RIGHT):
                            # SIDE_X -> SIDE_ZL   SIDE_Y -> SIDE_ZR
                            side ^= VOX_SIDE_BOTTOM >> VOX_SIDE_SHIFT

                    for xo, yo in SIDE_PIXELS[side]:
                        data[yy + yo, xx + xo] = colour
        else:
            raise ValueError(f'Requested unsuported vox rendering zoom: {zoom}')

        im = Image.fromarray(data, mode='P')
        im.putpalette(PALETTE)
        return ImageSprite(im, zoom=zoom, **kw)


class VoxTrainFile:

    class PreciseView:
        def __init__(self, x_axis, y_axis, z_axis):
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.z_axis = z_axis
            self.data = None

        def load(self, reader, palette):
            S = 4
            S2 = S * 2
            S4 = S * 4
            XY = 15 #7
            stretched_x = self.x_axis.copy()
            stretched_x[1] *= 1.5
            xx = reader.voxels @ stretched_x
            yy = reader.voxels @ self.y_axis
            zz = reader.voxels @ self.z_axis
            cc = reader.voxels[:, 3]

            xmin, xmax = np.amin(xx), np.amax(xx)
            ymin, ymax = np.amin(yy), np.amax(yy)
            self.w = int(xmax - xmin + 2 + .5) * S2
            self.h = int(ymax - ymin + 2 + .5) * S2
            im = Image.new('RGBA', (self.w, self.h), color=(0, 0, 0, 0))
            draw = ImageDraw.Draw(im)

            def cube(x, y, cx, cy, cz):
                x2 = x + S2
                x4 = x + S4
                y1 = y + S
                y2 = y + S2
                draw.polygon(((x, y1 - 1), (x2 - 2, y), (x2 + 1, y), (x4 - 1, y1 - 1), (x2 + 1, y2 - 2), (x2 - 2, y2 - 2)), fill=cz)
                draw.polygon(((x, y1), (x + 1, y1), (x2 - 1, y2 - 1), (x2 - 1, y2 - 1 + XY), (x2 - 2, y2 - 1 + XY), (x, y1 + XY)), fill=cx)
                draw.polygon(((x4 - 1, y1), (x4 - 2, y1), (x2, y2 - 1), (x2, y2 - 1 + XY), (x2 + 1, y2 - 1 + XY), (x4 - 1, y1 + XY)), fill=cy)


            for i in np.argsort(zz):
                x = int((xx[i] - xmin) * S2 + .5)
                y = int((yy[i] - ymin) * S2 + .5)
                c = cc[i]
                cube(
                    x, y,
                    tuple(palette[c | VOX_SIDE_X]),
                    tuple(palette[c | VOX_SIDE_Y]),
                    tuple(palette[c | VOX_SIDE_Z]),
                )

            origin = int(.5 - xmin) // 8, int(.5 - ymin) // 8
            # im.show()
            return im.resize((self.w // S2 // 8, self.h // S2 // 8), Image.BOX), origin


    class DiagView:
        def __init__(self, x_axis, y_axis, z_axis):
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.z_axis = z_axis
            self.data = None
            self.x_view = (self.y_axis[0] == 0)

        def load(self, reader, palette):
            xx = reader.voxels @ self.x_axis
            yy = reader.voxels @ self.y_axis
            zz = reader.voxels @ self.z_axis
            cc = reader.voxels[:, 3]

            xmin, xmax = np.amin(xx), np.amax(xx)
            ymin, ymax = np.amin(yy), np.amax(yy)
            # print(reader.size)
            # reader.size[2]
            self.w = xmax - xmin + 1
            h0 = self.h = ymax - ymin + 3
            self.w = (self.w + 3) // 4 * 4
            self.h = (self.h + 7) // 4 * 4
            #ymin += self.h - h0 # - 3 * (not self.x_view)

            zbuf = np.full((self.h, self.w), -1e100)
            data = np.zeros((self.h, self.w), dtype=np.uint16)

            def set_pixel(x, y, z, c):
                if zbuf[y, x] >= z:
                    return
                zbuf[y, x] = z
                data[y, x] = c

            for x, y, z, c in zip(xx, yy, zz, cc):
                x -= xmin
                y = int(y - ymin + .5)
                set_pixel(x, y, z, c | VOX_SIDE_Z)
                set_pixel(x, y + 1, z, c | VOX_SIDE_XY)
                set_pixel(x, y + 2, z, c | VOX_SIDE_XY)

            data = palette[data]
            im = Image.fromarray(data, mode='RGBA')
            im = im.resize((im.size[0] // 4, im.size[1] // 8), Image.BOX)
            origin = -xmin // 4, int(.5 - ymin) // 8
            return im, origin

    def __init__(self, path, *,
                 xy_colour_func=lambda c: c.darken(20),
                 x_colour_func=lambda c: c.darken(40),
                 y_colour_func=None,
                 z_colour_func=lambda c: c.brighten(10)):
        if isinstance(path, VoxReader):
            self.reader = path
            self.path = None
        else:
            self.path = path
            self.reader = VoxReader(path)

        self.xy_colour_func = xy_colour_func
        self.x_colour_func = x_colour_func
        self.y_colour_func = y_colour_func
        self.z_colour_func = z_colour_func

        self.data = None

    def get_image(self):
        pass

    def _make_palette(self, reader):
        def make_remap(func):
            if func is None:
                return reader.vox_palette
            l = map(func, reader.palette)
            palette = np.array(tuple(x.rgb for x in l))
            palette[palette > 1.] = 1.
            palette = np.rint(palette * 255).astype(np.uint8)
            transparency = reader.vox_palette[:,3]
            transparency.shape = (len(transparency), 1)
            palette = np.concatenate((palette, transparency), axis=1)
            return palette

        REMAP_XY = make_remap(self.xy_colour_func)
        REMAP_X = make_remap(self.x_colour_func)
        REMAP_Y = make_remap(self.y_colour_func)
        REMAP_Z = make_remap(self.z_colour_func)
        palette = np.concatenate((REMAP_Z, REMAP_XY, REMAP_X, REMAP_Y))
        palette[0] = (0, 0, 0, 0)

        return palette.astype(np.uint8)

    def _debug_sprites(self, sprites):
        rx, ry = 0, 0

        for s in sprites:
            rx += s.w
            ry = max(ry, s.h)
        im = Image.new('RGBA', (rx + 10 * len(sprites) - 10, ry))
        x = 0
        for s in sprites:
            im.paste(s.get_image()[0], (x, 0))
            x += s.w + 10
        im = im.resize((im.size[0] * 10, im.size[1] * 10), Image.NEAREST)
        im.show()

    def make_sprites(self):
        self.reader.read()
        palette = self._make_palette(self.reader)

        ROTATE = np.array((
            ( 0, 1, 0, 0),
            (-1, 0, 0, 0),
            ( 0, 0, 1, 0),
            ( 0, 0, 0, 1)
        ), dtype=int)
        x = np.array((1, -1, 0, 0), dtype=float)
        y = np.array((.5, .5, -2, 0), dtype=float)
        z = np.array((1., 1, 2 / 3**.5, 0))

        sprites = []

        def add_sprite(v):
            sprites.append(v.load(self.reader, palette))

        for i in range(4):
            add_sprite(self.PreciseView(x, y, z))
            x = x @ ROTATE
            y = y @ ROTATE
            z = z @ ROTATE

        add_sprite(self.DiagView(
            np.array((1, 0, 0, 0), dtype=int),
            np.array((0, 1, -2, 0), dtype=int),
            np.array((2, 0, 1, 0), dtype=int),
        ))
        add_sprite(self.DiagView(
            np.array((-1, 0, 0, 0), dtype=int),
            np.array((0, 1, -2, 0), dtype=int),
            np.array((-2, 0, 1, 0), dtype=int),
        ))
        add_sprite(self.DiagView(
            np.array((0, -1, 0, 0), dtype=int),
            np.array((1, 0, -2, 0), dtype=int),
            np.array((1, 0, 1, 0), dtype=int),
        ))
        add_sprite(self.DiagView(
            np.array((0, 1, 0, 0), dtype=int),
            np.array((1, 0, -2, 0), dtype=int),
            np.array((1, 0, 1, 0), dtype=int),
        ))

        def make_sprite(i, xofs, yofs):
            im, (ox, oy) = sprites[i]
            # return ImageSprite(im, xofs=ox + xofs, yofs=oy + yofs)
            return ImageSprite(im, xofs=xofs - ox, yofs=yofs - oy)

        # sprites = [sprites[x] for x in (7, 1, 4, 0, 6, 3, 5, 2)]
        res = (
            # 4 / 8 (+ox)
            # make_sprite(7, 1, -14),  # |
            # make_sprite(1, 4, -24),  # /
            # make_sprite(4, 11, 16),  # -
            # make_sprite(0, -1, -10),  # \
            # make_sprite(6, -21, -6),  # |
            # make_sprite(3, -34, -16),  # /
            # make_sprite(5, -41, -16),  # -
            # make_sprite(2, -17, -29),  # \
            make_sprite(7, -26, -14),  # |
            make_sprite(1, -30, 4),  # /
            make_sprite(4, -27, -11),  # -
            make_sprite(0, 3, -10),  # \
            make_sprite(6, 28, -8),  # |
            make_sprite(3, 24, 4),  # /
            make_sprite(5, 26, -11),  # -
            make_sprite(2, 3, 17),  # \
        )
        # self._debug_sprites(res)
        return res
