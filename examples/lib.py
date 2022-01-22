import grf


class RoadVehicle(grf.SpriteGenerator):
    class Flags:
        USE_2CC = 2

    def __init__(self, *, id, name, sprites, **props):
        if len(sprites) != 8:
            raise ValueError(f'RoadVehicle expects 8 sprites, found {len(sprites)}')
        self.id = id
        self.sprites = sprites
        self.name = name
        self.props = props

    def get_sprites(self):
        return [
            grf.Action4(
                feature=grf.RV,
                offset=self.id,
                is_generic_offset=False,
                strings=[self.name.encode('utf-8')]
            ),
            grf.Action0(
                feature=grf.RV,
                first_id=self.id,
                count=1,
                props={
                    'sprite_id': 0xff,
                    **self.props
                }
            ),
            grf.Action1(
                feature=grf.RV,
                set_count=1,
                sprite_count=8,
            ),
            *self.sprites,
            grf.GenericSpriteLayout(
                feature=grf.RV,
                ref_id=0,
                ent1=(0,),
                ent2=(0,),
            ),
            grf.Action3(
                feature=grf.RV,
                ids=[self.id],
                maps=[],
                default=grf.Ref(0),
            ),
        ]
