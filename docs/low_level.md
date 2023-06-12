# Low-level API
Closely mimics GRF/NFO capabilites.


## Feature
https://newgrf-specs.tt-wiki.net/wiki/Features

Actions 0-4 all take `feature` argument to specify OpenTTD feature they're working with.

Use one of the `TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, BRIDGE, HOUSE, GLOBAL_VAR, INDUSTRY_TILE, INDUSTRY, CARGO, SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, AIRPORT_TILE, ROADTYPE, TRAMTYPE` predefined constants of a `Feature` type.

For some actions `feature` argument is optional, but in that case they must be referenced by another action with a defined feature.


## Ref ID

Some actions take `ref_id` argument that can be used to reference them from other actions (in GRF specs it's called `set-id`). You can either specify it manually or just let grf-py set the ID for you. When referencing another action you can either use `Ref(id)` object to reference it by the ID or store that action in a variable and use it directly.


## Action 0
https://newgrf-specs.tt-wiki.net/wiki/Action0
[Documentation on properties](entity_properties.md).

### Define(*, feature, id, props)
- `id` - The ID of entity to define.
- `props` - dict of entity properties.

### DefineMultiple(*, feature, first_id, props, count=None)
- `first_id` - The ID of first entity to define.
- `props` - Dict of lists of properties to change. Each list should contain `count` elements.
- `count` - Amount of entities to define. If None the length of props lists is used.


## Action 1

https://newgrf-specs.tt-wiki.net/wiki/Action1

### SpriteSet(feature, count)

- `count` - Amount of sprites.

### Action1(feature, set_count, sprite_count)

- `set_count` - Amount of sets of sprites.
- `sprite_count` - Amount of sprites in each set.


## Action 2
https://newgrf-specs.tt-wiki.net/wiki/Action2

<!-- ## GenericSpritelayout(*, ent1, ) -->
### BasicSpriteLayout(*, ground, building, feature=None, ref_id=None)
- `ground` - Ground `Sprite`.
- `buildigns` - List of building `Sprite`s.

### AdvancedSpriteLayout(*, ground, feature=None, ref_id=None, buildings=(), has_flags=True)
- `ground` - Ground `Sprite`.
- `buildigns` - List of building `Sprite`s.
- `has_flags` - TODO

### Switch(code, ranges, default, *, feature=None, ref_id=None, related_scope=False)
https://newgrf-specs.tt-wiki.net/wiki/VarAction2Advanced
Functional equivalent of nml switch statement.
- `code` - Code of the expression to evaluate. TODO language docs.
- `ranges` - Either list of `Range` instances or dict with keys being integers or tuples `(low, high)` and `Ref` or `CB` values. Integer value is the same as using `CB`.
- `default` - Default value to use if no ranges match.
- `related_scope` - Whether to resolve variable against current object or the related one.

### RandomSwitch(*, scope, triggers, cmp_all, lowest_bit, groups, count=None, feature=None, ref_id=None)
https://newgrf-specs.tt-wiki.net/wiki/RandomAction2
- `scope` - scope of the switch, either `'self'`, `'relative'`, or `parent`.
- `triggers` - TODO
- `cmp_all` - TODO
- `lowest_bit` - TODO
- `groups` - List of actions to choose from.
- `count` - TODO

### IndustryProductionCallback(inputs, outputs, do_again, version=2)
https://newgrf-specs.tt-wiki.net/wiki/Action2/Industries


## Action 3
https://newgrf-specs.tt-wiki.net/wiki/Action3
### Action3(*, feature, ids, maps, default, wagon_override)
- `ids` - ID of the entities to associate graphics with
- `maps` - cargo type to ref ID dict
- `default` - default ref ID

### Map(definition, maps, default)
- `definition` - entity (action 0) to map graphics for
- `maps` - cargo type to ref ID dict
- `default` - default ref ID


## Action 4
https://newgrf-specs.tt-wiki.net/wiki/Action4

### DefineStrings(*, feature, offset, is_generic_offset, strings, lang=ANY_LANGUAGE)
- `offset` - ID of the first string to change
- `is_generic_offset` - TODO (0x80 in lang)
- `strings` - list of strings
- `lang` - language indentifier


## Action 5
https://newgrf-specs.tt-wiki.net/wiki/Action5

### ReplaceNewSprites(set_type, count, *, offset=None)
- `set_type` - type of the graphics.
- `count` - number of sprites to replace.
- `offset` - offset in the assignable block of sprites (to replace only a subset of sprites).


## Action 6
https://newgrf-specs.tt-wiki.net/wiki/Action6
### ModifySprite(params)
- `params` - list of dicts with param info with keys 'num', 'size', 'offset'.


## Action 7 and 9
https://newgrf-specs.tt-wiki.net/wiki/Action7
### If(*, is_static, variable, condition, value, skip, varsize=None)
- `is_static` - True for action 9, False for 7.
- `variable` - variable to base the decision on.
- `condition` - condition type.
- `value` - what the variable is compared to.
- `skip` - how many sprites will be skipped.
- `varsize` - TODO


## Action 8
https://newgrf-specs.tt-wiki.net/wiki/Action8

### SetDescription(*, grfid, name, description, format_version=8)
- `grfid` - a unique ID for your .grf file (4 bytes).
- `name` - name of the GRF.
- `description`
- `format_version` - GRF format version (int).


## Action A
https://newgrf-specs.tt-wiki.net/wiki/ActionA
### ReplaceOldSprites(sets)
- `sets` (list, tuple) -  a list of pairs (first, num) defining the sets of sprites to replace


## Action B
https://newgrf-specs.tt-wiki.net/wiki/ActionB
### ErrorMessage(*, severity, lang, message, text_param=None, params=None)
- `severity`
- `lang`
- `message`
- `text_param`
- `params`


## Action C
https://newgrf-specs.tt-wiki.net/wiki/ActionC
### Comment(data)
- `data` - content of the comment, has no effect on the GRF functionality.


## Action D
https://newgrf-specs.tt-wiki.net/wiki/ActionD
### ComputeParameters(target, operation, if_undefinde, source1, source2, value=None)


## Action E
https://newgrf-specs.tt-wiki.net/wiki/ActionE
Unsupported (TODO)


## Action F
https://newgrf-specs.tt-wiki.net/wiki/ActionF
Unsupported (TODO)


## Action 10
https://newgrf-specs.tt-wiki.net/wiki/Action10
### Label(label, comment)
- `label` (int 0..255) - ID of the label
- `comment` (bytes) - optional comment, ignored by the game


## Action 11
https://newgrf-specs.tt-wiki.net/wiki/Action11
### SoundEffects(count)
- `count` - number of sound effects that follow

### ImportSound(grfid, number)
Special action that can follow `SoundEffects` to refer to a sound in another GRF.


## Action 12
https://newgrf-specs.tt-wiki.net/wiki/Action12
### UnicodeGlyphs(glyphs)
- `glyphs` - List of dicts with 'font', 'count' and 'base' keys.


## Action 13
https://newgrf-specs.tt-wiki.net/wiki/Action13
### Translations(*, grfid, offset, is_generic_offset, lang=ANY_LANGUAGE)


## Action 14
https://newgrf-specs.tt-wiki.net/wiki/Action14
### SetProperties(props)
- `props` (dict) - Nested dicts of properties. All keys must be 4 bytes, values either int, bytes, string, tuple or another dict. If the value is tuple it must be a pair of `(lang_id, str)`.
