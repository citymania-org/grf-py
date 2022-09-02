import struct
import numpy as np
from PIL import Image

from .common import to_spectra, ZOOM_4X, ZOOM_2X, ZOOM_NORMAL, PALETTE
from .sprites import ImageSprite, convert_image, PaletteRemap, find_best_color


VOX_SIDE_RIGHT, VOX_SIDE_BOTTOM = 0x100, 0x200
VOX_SIDE_SHIFT, VOX_SIDE_MASK = 8, VOX_SIDE_RIGHT | VOX_SIDE_BOTTOM
VOX_SIDE_ZL, VOX_SIDE_ZR, VOX_SIDE_X, VOX_SIDE_Y = 0, VOX_SIDE_RIGHT, VOX_SIDE_BOTTOM, VOX_SIDE_RIGHT | VOX_SIDE_BOTTOM


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
        self.voxels[:, [0, 1]] = self.voxels[:, [1, 0]]
        self.voxels[:, 0] = -self.voxels[:, 0]
        self.voxels = np.array((self.size[0], 0, 0, 0)) + self.voxels

    def _read_rgba(self, f):
        vox_palette = np.fromfile(f, count=256 * 4, dtype=np.uint8)
        vox_palette.shape = (256, 4)
        self.palette = [find_best_color(to_spectra(r, g, b)) for r, g, b, _ in vox_palette]

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
        cc = np.take(self.reader.palette, self.reader.voxels[:, 3])
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
        return ImageSprite(im, data.shape[1], data.shape[0], zoom=zoom, **kw)
