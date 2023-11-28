import grf
from .vehicle import Vehicle


SHIP_KMH_TO_NFO = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508]
SHIP_MPH_TO_NFO = [0, 4, 8, 10, 14, 16, 20, 24, 26, 30, 32, 36, 40, 42, 46, 48, 52, 56, 58, 62, 64, 68, 72, 74, 78, 80, 84, 88, 90, 94, 96, 100, 104, 106, 110, 112, 116, 120, 122, 126, 128, 132, 136, 138, 142, 144, 148, 152, 154, 158, 160, 164, 168, 170, 174, 176, 180, 184, 186, 190, 192, 196, 200, 202, 206, 208, 212, 216, 218, 222, 224, 228, 232, 234, 238, 240, 244, 248, 250, 254, 256, 260, 264, 266, 270, 272, 276, 280, 282, 286, 288, 292, 296, 298, 302, 304, 308, 312, 314, 318, 320, 324, 328, 330, 334, 336, 340, 344, 346, 350, 352, 356, 360, 362, 366, 368, 372, 376, 378, 382, 384, 388, 392, 394, 398, 400, 404, 408, 410, 414, 416, 420, 424, 426, 430, 432, 436, 440, 442, 446, 448, 452, 456, 458, 462, 464, 468, 472, 474, 478, 480, 484, 488, 490, 494, 496, 500, 504, 506, 510, 512, 516, 520, 522, 526, 528, 532, 536, 538, 542, 544, 548, 552, 554, 558, 560, 564, 568, 570, 574, 576, 580, 584, 586, 590, 592, 596, 600, 602, 606, 608, 612, 616, 618, 622, 624, 628, 632, 634, 638, 640, 644, 648, 650, 654, 656, 660, 664, 666, 670, 672, 676, 680, 682, 686, 688, 692, 696, 698, 702, 704, 708, 712, 714, 718, 720, 724, 728, 730, 734, 736, 740, 744, 746, 750, 752, 756, 760, 762, 766, 768, 772, 776, 778, 782, 784, 788, 792, 794, 798, 800, 804, 808, 810, 814, 816]

def _check_speed(speed, array, unit):
    if speed < 0:
        raise ValueError("Speed can't be negative.")
    if speed > len(array):
        raise ValueError(f'Speeds over {len(array)} {unit} are not supported.')
    return array[speed]


class Ship(Vehicle):
    Flags = grf.ShipFlags

    @staticmethod
    def kmh(speed):
        return _check_speed(speed, SHIP_KMH_TO_NFO, 'km/h')

    @staticmethod
    def kmhish(speed):
        return speed // 2

    @staticmethod
    def mph(speed):
        # TODO probably doesn't show exact mph in the game
        return _check_speed(speed, SHIP_MPH_TO_NFO, 'mph')

    def __init__(self, *, id, name, max_speed, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None, **props):
        super().__init__(grf.SHIP, liveries=liveries, callbacks=callbacks, purchase_sprite=purchase_sprite, additional_text=additional_text)

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self._props = props

    # def _set_callbacks(self, g):
    #     res = super()._set_callbacks(g)

    #     if self.max_speed.precise_value >= 0x400:
    #         # TODO only set speed property
    #         self.callbacks.change_properties = grf.Switch(
    #             ranges={
    #                 0x15: self.max_speed.value,
    #             },
    #             default=self.callbacks.graphics,
    #             code='extra_callback_info1_byte',
    #         )

    #     return res

    def get_sprites(self, g):
        res = self._gen_purchase_sprites()

        if self.liveries:
            layouts = []
            for i, l in enumerate(self.liveries):
                layouts.append(grf.GenericSpriteLayout(
                    ent1=(i,),
                    ent2=(i,),
                ))

            if len(self.liveries) > 1:

                self.callbacks.graphics.default = grf.Switch(
                    related_scope=True,
                    ranges=dict(enumerate(layouts)),
                    default=layouts[0],
                    code='cargo_subtype',
                )
            else:
                self.callbacks.graphics.default = layouts[0]

        res.extend(self._set_callbacks(g))

        # Liveries
        if self.liveries and len(self.liveries) > 1:
            self.callbacks.cargo_subtype_text = grf.Switch(
                ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
                default=0x400,
                code='cargo_subtype',
            )

        self.callbacks.set_flag_props(self._props)

        res.extend(self._gen_name_sprites(g, self.id))

        res.append(definition := grf.Define(
            feature=grf.SHIP,
            id=self.id,
            props={
                'sprite_id': 0xff,
                'max_speed': self.max_speed,
                **self._props
            }
        ))
        if self.liveries:
            res.append(grf.Action1(
                feature=grf.SHIP,
                set_count=len(self.liveries),
                sprite_count=8,
            ))

            for l in self.liveries:
                res.extend(l['sprites'])

        res.extend(self.callbacks.make_map_action(definition))
        return res
