import grf

from .callback import make_callback_manager

class Badge(grf.SpriteGenerator):
    
    class Flags():
        COPY_TO_RELATED_ENTITY = 0x01
        NAME_LIST_STOP = 0x02
        NAME_LIST_FIRST_ONLY = 0x04
        USE_COMPANY_COLOUR = 0x08
        NAME_SKIP = 0x16
    
    def __init__(self, label, name=None, icon=None, **props):
        self.name = name
        self.label = label
        self.icon = icon
        self._props = props
        
    def _gen_name_sprites(self, g, bid):
        if isinstance(self.name, grf.StringRef):
            return self.name.get_actions(self.feature, bid)
        else:
            return g.strings.add(self.name).get_actions(grf.BADGE, bid)
        
    def _make_graphics(self):
        res = [grf.Action1(grf.BADGE, 1, 1)]
        res.append(self.icon)
        sl = grf.GenericSpriteLayout(ent1=(0,), ent2=(0,))
        return res, sl
    
    def get_sprites(self, g):
        
        if self.icon is not None:
            self.callbacks = make_callback_manager(grf.BADGE, None)
            res, self.callbacks.graphics.default = self._make_graphics()
        else:
            res = []
        
        bid = g.resolve_id(grf.BADGE, self.label)
        
        res.append(definition := grf.Define(
            feature=grf.BADGE,
            id=bid,
            props={
                'label': self.label,
                **self._props
            }
        ))
        res.extend(self._gen_name_sprites(g, bid))
        
        if self.icon is not None:
            res.extend(self.callbacks.make_map_action(definition))
        
        return res