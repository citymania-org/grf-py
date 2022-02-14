import struct
import numpy as np
from PIL import Image

from .grf import BaseSprite, ZOOM_4X, convert_image, BPP_8, BPP_24, BPP_32, RealSprite, fix_palette


class NMLFileSpriteWrapper:
    def __init__(self, sprite, file, bit_depth):
        self.sprite = sprite
        self.file = type('Filename', (object, ), {'value': file.path})
        self.bit_depth = bit_depth
        self.mask_file = type('Filename', (object, ), {'value': file.mask}) if file.mask else None
        self.mask_pos = None
        self.flags = type('Flags', (object, ), {'value': 0})

    xpos = property(lambda self: type('XSize', (object, ), {'value': self.sprite.x}))
    ypos = property(lambda self: type('YSize', (object, ), {'value': self.sprite.y}))

    xsize = property(lambda self: type('XSize', (object, ), {'value': self.sprite.w}))
    ysize = property(lambda self: type('YSize', (object, ), {'value': self.sprite.h}))

    xrel = property(lambda self: type('XSize', (object, ), {'value': self.sprite.xofs}))
    yrel = property(lambda self: type('YSize', (object, ), {'value': self.sprite.yofs}))


class BaseImageSprite(RealSprite):
    def __init__(self, x, y, w, h, *, bpp=None, mask=None, **kw):
        if bpp == BPP_8 and mask is not None:
            raise ValueError("8bpp sprites can't have a mask")
        super().__init__(w, h, **kw)
        self.bpp = bpp
        self.mask = mask
        self.x = x
        self.y = y
        self.name = f'{self.x},{self.y} {self.w}x{self.h}'

    def get_image(self):
        raise NotImplementedError

    def get_mask_image(self):
        raise NotImplementedError

    def get_colourkey(self):
        return None

    # https://github.com/OpenTTD/grfcodec/blob/master/docs/grf.txt
    def get_real_data(self, encoder):
        img, bpp = self.get_image()
        if self.bpp is None:
            self.bpp = bpp
        npalpha = None

        if bpp != self.bpp:
            print(f'Sprite {self.name} expected {self.bpp}bpp but file is {bpp}bpp, converting.')
            if self.bpp == BPP_24:
                img = img.convert('RGB')
            elif self.bpp == BPP_32:
                img = img.convert('RGBA')
            elif self.bpp == BPP_8:
                p_img = Image.new('P', (16, 16))
                p_img.putpalette(PALETTE)
                img = img.quantize(palette=p_img, dither=0)
        else:
            if bpp == BPP_8:
                img = fix_palette(img, self.name)

        npimg = np.asarray(img)
        colourkey = self.get_colourkey()
        if colourkey is not None:
            if self.bpp == BPP_24:
                npalpha = 255 * np.all(np.not_equal(npimg, colourkey), axis=2)
                npalpha = npalpha.astype(np.uint8, copy=False)
            elif self.bpp == BPP_32:
                if len(colourkey) == 3:
                    colourkey = (*colourkey, 255)
                npalpha = 255 * np.all(np.not_equal(npimg, colourkey), axis=2)
                npalpha = npalpha.astype(np.uint8, copy=False)
            else:
                print(f'Colour key on 8bpp sprites is not supported')

        info_byte = 0x40
        if self.bpp == BPP_24 or self.bpp == BPP_32:
            stack = [npimg]

            if self.bpp == BPP_32:
                npimg.shape = (self.w * self.h, 4)
                rbpp = 4
                info_byte |= 0x1 | 0x2  # rgb, alph

                if npalpha is not None:
                    npimg = npimg[:, :3]
                    npalpha.shape = (self.w * self.h, 1)
                    stack = [npimg, npalpha]
            else:
                npimg.shape = (self.w * self.h, 3)
                rbpp = 3
                info_byte = 0x1  # rgb

                if npalpha is not None:
                    npalpha.shape = (self.w * self.h, 1)
                    stack.append(npalpha)
                    rbpp += 1
                    info_byte |= 0x2  # alpha


            if self.mask is not None:
                mask_file, xofs, yofs = self.mask
                mask_img, mask_bpp = self.get_mask_image()
                if mask_bpp != BPP_8:
                    raise RuntimeError(f'Mask {mask_file.path} is not an 8bpp image')
                xofs, yofs = self.x + xofs, self.y + yofs
                mask_img = mask_img.crop((xofs, yofs, xofs + self.w, yofs + self.h))
                mask_img = fix_palette(mask_img, f'{mask_file.path} {xofs},{yofs} {self.w}x{self.h}')
                npmask = np.asarray(mask_img)
                npmask.shape = (self.w * self.h, 1)
                stack.append(npmask)
                rbpp += 1
                info_byte |= 0x4  # mask

            if len(stack) > 1:
                raw_data = np.concatenate(stack, axis=1)
            else:
                raw_data = stack[0]
            raw_data.shape = self.w * self.h * rbpp

        else:
            info_byte |= 0x4  # pal/mask
            raw_data = npimg
            raw_data.shape = self.w * self.h

        data = encoder.sprite_compress(raw_data)
        return struct.pack(
            '<IIBBHHhh',
            self.sprite_id,
            len(data) + 10, info_byte,
            self.zoom,
            self.h,
            self.w,
            self.xofs,
            self.yofs,
        ) + data


class ImageSprite(BaseImageSprite):
    def __init__(self, image, x, y, w, h, *, mask=None, **kw):
        self._image = convert_image(image)
        if mask:
            mask_image, xofs, yofs = mask
            mask = convert_image(mask_image), xofs, yofs
        super().__init__(x, y, w, h, bpp=self._image[1], mask=mask, **kw)

    def get_image(self):
        return self._image

    def get_mask_image(self):
        return self.mask[0] if self.mask else None


class ImageFile:
    def __init__(self, path, colourkey=None):
        self.path = path
        self.colourkey = colourkey
        self._image = None

    def get_image(self):
        if self._image:
            return self._image
        img = Image.open(self.path)
        if img.mode == 'P':
            self._image = (img, BPP_8)
        elif img.mode == 'RGB':
            self._image = (img, BPP_24)
        else:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            self._image = (img, BPP_32)
        return self._image


class FileSprite(BaseImageSprite):
    def __init__(self, file, x, y, w, h, *, bpp=None, mask=None, **kw):
        assert(isinstance(file, ImageFile))
        super().__init__(x, y, w, h, bpp=bpp, mask=mask, **kw)
        self.file = file

    def get_image(self):
        img, bpp = self.file.get_image()
        img = img.crop((self.x, self.y, self.x + self.w, self.y + self.h))
        return img, bpp

    def get_mask_image(self):
        mask_file, _, _ = self.mask
        return mask_file.get_image()

    def get_colourkey(self):
        return self.file.colourkey

