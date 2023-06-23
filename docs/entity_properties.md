# Define/DefineMultiple (Action 0) properties
Closely match NFO properties of the feature. Wherever makes sense use the same name as equivalent NML properties.


## Common vehicle properties
https://newgrf-specs.tt-wiki.net/wiki/Action0/Vehicles

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x00 | introduction_days_since_1920 | int(0-65535) | deprecated by `introduction_date`, date of introduction as the amount of days since 1920 |
| 0x02 | reliability_decay | int(0-255) | reliability decay speed |
| 0x03 | vehicle_life | int(0-255) | vehicle life in years |
| 0x04 | model_life | int(0-255) | model life in years |
| 0x06 | climates_available | ClimateProperty | climate availability |
| 0x07 | loading_speed | int(0-255) | loading speed |


## Trains (TRAIN, 0x00)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Vehicles/Trains

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x05 | track_type | int(0-255) | Track type |
| 0x08 | ai_special_flag | int(0-255) | AI special flag: set to 1 if engine is 'optimized' for passenger service (AI won't use it for other cargo), 0 otherwise |
| 0x09 | max_speed | int(0-255) | Speed as mph*1.6 |
| 0x0B | power | int(0-65535) | Power (0 for wagons) |
| 0x0D | running_cost_factor | int(0-255) | Running cost factor (0 for wagons) |
| 0x0E | running_cost_base | int(0-4294967295) | Running cost base |
| 0x12 | sprite_id | int(0-255) | Sprite ID (`0xFD` for new graphics) |
| 0x13 | dual_headed | int(0-255) | Dual-headed flag; 1 if dual-headed engine, 0 otherwise |
| 0x14 | cargo_capacity | int(0-255) | Cargo capacity |
| 0x15 | default_cargo_type | int(0-255) | Cargo type |
| 0x16 | weight_low | int(0-255) | Weight in tons |
| 0x17 | cost_factor | int(0-255) | Cost factor |
| 0x18 | ai_engine_rank | int(0-255) | Engine rank for the AI (AI selects the highest-rank engine of those it can buy) |
| 0x19 | engine_class | int(0-255) | Engine traction type |
| 0x1A | sort_purchase_list | int(0-65535) | Not a property, but an action: sort the purchase list |
| 0x1B | extra_power_per_wagon | int(0-65535) | Power added by each wagon connected to this engine |
| 0x1C | refit_cost | int(0-255) | Refit cost, using 50% of the purchase price cost base |
| 0x1D | refittable_cargo_types | int(0-4294967295) | Bit mask of cargo types available for refitting, see column 2 (bit value) in CargoTypes |
| 0x1E | cb_flags | int(0-255) | Callback flags bit mask, see below  yes |
| 0x1F | tractive_effort_coefficient | int(0-255) | Coefficient of tractive effort |
| 0x20 | air_drag_coefficient | int(0-255) | Coefficient of air drag |
| 0x21 | shorten_by | int(0-255) | Make vehicle shorter by this amount |
| 0x22 | visual_effect_and_powered | int(0-255) | Set visual effect type (steam/smoke/sparks) as well as position |
| 0x23 | extra_weight_per_wagon | int(0-255) | Set how much weight is added by making wagons powered (i.e. weight of engine) |
| 0x24 | weight_high | int(0-255) | High byte of vehicle weight, weight will be prop.24*256+prop.16 |
| 0x25 | bitmask_vehicle_info | int(0-255) | User-defined bit mask to set when checking veh. var. 42 |
| 0x26 | retire_early | int(-128-127) | Retire vehicle early, this many years before the end of phase 2 (see Action0General) |
| 0x27 | misc_flags | int(0-255) | Miscellaneous flags |
| 0x28 | refittable_cargo_classes | int(0-65535) | Refittable cargo classes |
| 0x29 | non_refittable_cargo_classes | int(0-65535) | Non-refittable cargo classes |
| 0x2A | introduction_date | DateProperty | Long format introduction date|
| 0x2B | cargo_age_period | int(0-65535) | period |
| 0x2C | cargo_allow_refit | bytes | refittable cargo types |
| 0x2D | cargo_disallow_refit | bytes | refittable cargo types |
| 0x2E | curve_speed_mod | int(0-65535) | speed modifier |
| 0x2F | variant_group | int(0-65535) | Vehicle variant group |
| 0x30 | extra_flags | int(0-4294967295) | extra flags |


## Road vehicles (RV, 0x01)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Vehicles/RoadVehicles

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x05 | road_type | int(0-255) | Roadtype / tramtype (see below) |
| 0x08 | precise_max_speed | int(0-255) | Speed as mph*3.2 |
| 0x09 | running_cost_factor | int(0-255) | Running cost factor |
| 0x0A | running_cost_base | int(0-4294967295) | Running cost base |
| 0x0E | sprite_id | int(0-255) | Sprite ID (`0xFF` for new graphics) |
| 0x0F | cargo_capacity | int(0-255) | Capacity |
| 0x10 | default_cargo_type | int(0-255) | Cargo type |
| 0x11 | cost_factor | int(0-255) | Cost factor |
| 0x12 | sound_effect | int(0-255) | Sound effect: 0x17/0x19/0x1A for regular, 0x3C/0x3E for toyland |
| 0x13 | power | int(0-255) | Power in 10 hp |
| 0x14 | weight | int(0-255) | Weight in 1/4 tons |
| 0x15 | max_speed | int(0-255) | Speed as mph*0.8 |
| 0x16 | refittable_cargo_types | int(0-4294967295) | Bit mask of cargo types available for refitting (not refittable if 0 or unset) |
| 0x17 | cb_flags | int(0-255) | Callback flags bit mask |
| 0x18 | tractive_effort_coefficient | int(0-255) | Coefficient of tractive effort |
| 0x19 | air_drag_coefficient | int(0-255) | Coefficient of air drag |
| 0x1A | refit_cost | int(0-255) | Refit cost, using 25% of the purchase price cost base |
| 0x1B | retire_early | int(-128-127) | Retire vehicle early, this many years before the end of phase 2 |
| 0x1C | misc_flags | int(0-255) | Miscellaneous vehicle flags |
| 0x1D | refittable_cargo_classes | int(0-65535) | Refittable cargo classes, see train prop. 0x28 |
| 0x1E | non_refittable_cargo_classes | int(0-65535) | Non-refittable cargo classes, see train prop. 0x29 |
| 0x1F | introduction_date | DateProperty | Long format introduction date |
| 0x20 | sort_purchase_list | int(0-65535) | Sort the purchase list(action) |
| 0x21 | visual_effect | int(0-255) | Visual effect |
| 0x22 | cargo_age_period | int(0-65535) | Custom cargo ageing period |
| 0x23 | shorten_by | int(0-255) | Make vehicle shorter, see train property 0x21 |
| 0x24 | cargo_allow_refit | bytes | List of always refittable cargo types, see train property 0x2C |
| 0x25 | cargo_disallow_refit | bytes | List of never refittable cargo types, see train property 0x2D |
| 0x26 | variant_group | int(0-65535) | Vehicle variant group |
| 0x27 | extra_flags | int(0-4294967295) | Extra flags |


## Ships (SHIP, 0x02)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Vehicles/Ships

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | sprite_id | int(0-255) | Sprite (FF for new graphics) |
| 0x09 | is_refittable | int(0-255) | Refittable (0 no, 1 yes) |
| 0x0A | cost_factor | int(0-255) | Cost factor |
| 0x0B | speed | int(0-255) | Speed in mph*3.2 |
| 0x0C | cargo_type | int(0-255) | Cargo type, see CargoTypes |
| 0x0D | capacity | int(0-65535) | Capacity |
| 0x0F | running_cost_factor | int(0-255) | Running cost factor |
| 0x10 | sound | int(0-255) | Sound effect type (4=cargo ship, 5=passenger ship) |
| 0x11 | refittable_cargo_types | int(0-4294967295) | Bit mask of cargo types available for refitting, see column 2 (bit values) in CargoTypes |
| 0x12 | cb_flags | int(0-255) | Callback flags bit mask, see below |
| 0x13 | refit_cost | int(0-255) | Refit cost, using 1/32 of the default refit cost base |
| 0x14 | ocean_speed | int(0-255) | Ocean speed fraction, sets fraction of top speed available in the ocean; e.g. 00=100%, 80=50%, FF=0.4% |
| 0x15 | canal_speed | int(0-255) | Canal speed fraction, same as above but for canals and rivers |
| 0x16 | retire_early | int(-128-127) | Retire vehicle early, this many years before the end of phase 2 (see Action0General) |
| 0x17 | flags | int(0-255) | Miscellaneous vehicle flags |
| 0x18 | refit_classes | int(0-65535) | Refittable cargo classes, see train prop. 0x28 |
| 0x19 | non_refit_classes | int(0-65535) | Non-refittable cargo classes, see train prop. 0x29 |
| 0x1A | introduction_date | DateProperty | Long format introduction date |
| 0x1B | sort_purchase_list | int(0-255) | Sort the purchase list |
| 0x1C | visual_effect | int(0-255) | Visual effect |
| 0x1D | cargo_age_period | int(0-65535) | Custom cargo ageing period |
| 0x1E | cargo_allow_refit | bytes | List of always refittable cargo types, see train property 0x2C |
| 0x1F | cargo_disallow_refit | bytes | List of never refittable cargo types, see train property 0x2D |
| 0x20 | variant_group | int(0-65535) | Vehicle variant group |
| 0x21 | extra_flags | int(0-4294967295) | Extra flags |


# Aircraft (AIRCRAFT, 0x03)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Vehicles/Planes

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | sprite_id | int(0-255) | Sprite (`0xFF` for new graphics) |
| 0x09 | is_helicopter | int(0-255) | Is helicopter? 2=no, 0=yes |
| 0x0A | is_large | int(0-255) | Is large? 0=no, 1=yes (i.e. can't safely land on small airports) |
| 0x0B | cost_factor | int(0-255) | Cost factor |
| 0x0C | max_speed | int(0-255) | Speed in units of 8 mph, that is: property = (speed in mph) / 8 |
| 0x0D | acceleration | int(0-255) | Acceleration in units of 3/8 mph/tick, that is: property = (acceleration in mph/tick) * 8/3 [1] |
| 0x0E | running_cost_factor | int(0-255) | Running cost factor |
| 0x0F | passenger_capacity | int(0-65535) | Primary cargo capacity (passenger or refitted cargo) |
| 0x11 | mail_capacity | int(0-255) | Secondary cargo capacity (mail) |
| 0x12 | sound_effect | int(0-255) | Sound effect |
| 0x13 | refittable_cargo_types | int(0-4294967295) | Bit mask of cargo types available for refitting, see column 2 (Bit Value) in CargoTypes |
| 0x14 | cb_flags | int(0-255) | Callback flags bit mask, see below |
| 0x15 | refit_cost | int(0-255) | Refit cost, using 1/32 of the default refit cost base |
| 0x16 | retire_early | int(-128-127) | Retire vehicle early, this many years before the end of phase 2 (see Action0General) |
| 0x17 | misc_flags | int(0-255) | Miscellaneous vehicle flags |
| 0x18 | refittable_cargo_classes | int(0-65535) | Refittable cargo classes, see train prop. 28 |
| 0x19 | non_refittable_cargo_classes | int(0-65535) | Non-refittable cargo classes, see train prop. 29 |
| 0x1A | introduction_date | DateProperty | Long format introduction date |
| 0x1B | sort_purchase_list | int(0-255) | Sort the purchase list |
| 0x1C | cargo_age_period | int(0-65535) | Custom cargo ageing period |
| 0x1D | cargo_allow_refit | bytes | List of always refittable cargo types, see train property 2C |
| 0x1E | cargo_disallow_refit | bytes | List of never refittable cargo types, see train property 2D |
| 0x1F | range | int(0-65535) | Aircraft range in tiles. Distance is euclidean, a value of 0 means range is unlimited |
| 0x20 | variant_group | int(0-65535) | Vehicle variant group |
| 0x21 | extra_flags | int(0-4294967295) | extra flags |


# Stations (STATION, 0x04)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Stations

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | class | Label | Class ID |
| 0x09 | layout | OldStationLayoutProperty() | Sprite layout |
| 0x0A | copy_layout | int(0-255) | Copy sprite layout |
| 0x0B | cb_flags | int(0-255) | Callback flags |
| 0x0C | disabled_platforms | int(0-255) | Bit mask of disabled numbers of platforms |
| 0x0D | disabled_length | int(0-255) | Bit mask of disabled platform lengths |
| 0x0E | custom_layout | Not supported | Define custom layout |
| 0x0F | copy_custom_layout | int(0-255) | Copy custom layout from stationid given by argument |
| 0x10 | cargo_threshold | int(0-65535) | Little/lots threshold |
| 0x11 | draw_pylon_tiles | int(0-255) | Pylon placement |
| 0x12 | cargo_random_triggers | int(0-4294967295) | Bit mask of cargo type triggers for random sprites |
| 0x13 | general_flags | int(0-255) | General flags |
| 0x14 | hide_wire_tiles | int(0-255) | Overhead wire placement |
| 0x15 | non_traversable_tiles | int(0-255) | Can train enter tile |
| 0x16 | animation_info | int(0-65535) | Animation information |
| 0x17 | animation_speed | int(0-255) | Animation speed |
| 0x18 | animation_triggers | int(0-65535) | Animation triggers |
| 0x19 | road_routing | Not supported | Road routing (reserved for future use) |
| 0x1A | advanced_layout | NewStationLayoutProperty | Advanced sprite layout with register modifiers |
| 0x1B | min_bridge_height | 8 bytes (TODO) | dvanced sprite layout with register modifiers |


# Canals (CANAL, 0x05)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Canals

Not supported


# Bridges (BRIDGE, 0x06)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Bridges

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x00 | fallback_type | int(0-255) | Failback Type, a default TTD Bridge ID |
| 0x08 | intro_year_since_1920 | int(0-255) | Year of availability, counted from 1920 (set to 1920 for first bridge if newstartyear<1930) |
| 0x09 | min_length | int(0-255) | Minimum length, not counting ramps |
| 0x0A | max_length | int(0-255) | Maximum length, not counting ramps |
| 0x0B | cost_factor_byte | int(0-255) | Cost factor |
| 0x0C | max_speed | int(0-65535) | Max. speed |
| 0x0D | layout | dict (TODO) | Sprite layout, see below |
| 0x0E | flags | int(0-255) | Various flags, bitcoded |
| 0x0F | intro_year | int(0-4294967295) | Long format year of availability, counted from year 0 |
| 0x10 | purchase_text | int(0-65535) | Purchase text |
| 0x11 | description_rail | int(0-65535) | Description of a rail bridge |
| 0x12 | description_road | int(0-65535) | Description of a road bridge |
| 0x13 | cost_factor | int(0-65535) | Cost factor word access |


# Houses (HOUSE, 0x07)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Houses

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | substitute | int(0-255) | Substitute building type |
| 0x09 | flags | int(0-255) | Building flags |
| 0x0A | availability_years | int(0-65535) | Availability years |
| 0x0B | poopulation | int(0-255) |  Population |
| 0x0C | mail_mult | int(0-255) | Mail generation multiplier |
| 0x0D | passengers_acceptance | int(0-255) | Passenger acceptance |
| 0x0E | mail_acceptance | int(0-255) | Mail acceptance |
| 0x0F | goods_acceptance | int(0-255) | Goods, food or fizzy drinks acceptance |
| 0x10 | authority_impact | int(0-65535) | LA rating decrease on removal (should be set to the same value for every tile for multi-tile buildings |
| 0x11 | removal_cost_mult | int(0-255) | Removal cost multiplier (should be set to the same value for every tile for multi-tile buildings) |
| 0x12 | name | int(0-65535) | Building name ID |
| 0x13 | availability_mask | int(0-65535) | Building availability mask |
| 0x14 | cb_flags1 | int(0-255) | House callback flags |
| 0x15 | override | int(0-255) | House override byte |
| 0x16 | refresh_multiplier | int(0-255) | Periodic refresh multiplier |
| 0x17 | random_colours | 4 bytes (TODO) | Four random colours to use |
| 0x18 | probability | int(0-255) | Relative probability of appearing |
| 0x19 | extra_flags | int(0-255) | Extra flags  (high byte of prop 0x9) |
| 0x1A | animation_frames | int(0-255) | Animation frames |
| 0x1B | animation_speed | int(0-255) | Animation speed |
| 0x1C | building_class | int(0-255) | Class of the building type |
| 0x1D | cb_flags2 | int(0-255) | Callback flags 2 |
| 0x1E | accepted_cargo | int(0-4294967295) | Accepted cargo types |
| 0x1F | min_life | int(0-65535) | Minimum life span in years |
| 0x20 | watched_cargo_types | int(0-65535) | Cargo acceptance watch list |
| 0x21 | min_year | int(0-65535) | Long year (zero based) of minimum appearance |
| 0x22 | max_year | int(0-65535) | Long year (zero based) of maximum appearance |
| 0x23 | tile_acceptance | Not supported | Tile acceptance list |


# Global settings (GLOBAL_VAR, 0x08)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Global_Settings

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | basecost | int(0-255) | Cost base multipliers |
| 0x09 | cargo_table | Label | Cargo translation table |
| 0x0A | currency_name | int(0-65535) | Currency display names |
| 0x0B | currency_mult | int(0-4294967295) | Currency multipliers |
| 0x0C | currency_options | int(0-65535) | Currency options |
| 0x0D | currency_symbols | 0E | Currency symbols |
| 0x0F | currency_euro_date | int(0-65535) | Euro introduction dates |
| 0x10 | snowline_table | 12*32*B | Snow line height table |
| 0x11 | grfid_overrides | 2*Label | GRFID overrides for engines |
| 0x12 | railtype_table | Label | Railtype translation table |
| 0x13 | lang_genders | MultiDictProperty |  Gender/case translation table |
| 0x14 | lang_cases | MultiDictProperty |  Gender/case translation table |
| 0x15 | lang_plural | int(0-255) | Plural form |
| 0x16 | roadtype_table1 | int(0-255) | Road-/tramtype translation table |
| 0x17 | roadtype_table2 | int(0-4294967295) | Road-/tramtype translation table |


# Industry tiles (INDUSTRY_TILE, 0x09)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Industry_Tiles

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | building_type | int(0-255) | Substitute building type |
| 0x09 | tile_override | int(0-255) | Industry tile override |
| 0x0A | tile_acceptance_1 | int(0-65535) | Tile acceptance |
| 0x0B | tile_acceptance_2 | int(0-65535) | Tile acceptance |
| 0x0C | tile_acceptance_3 | int(0-65535) | Tile acceptance |
| 0x0D | land_shape_flags | int(0-255) | Land shape flags |
| 0x0E | cb_flags | int(0-255) | Callback flags |
| 0x0F | anim_info | int(0-65535) | Animation information |
| 0x10 | anim_speed | int(0-255) | Animation speed. |
| 0x11 | cb25_triggers | int(0-255) | Triggers for callback 25 |
| 0x12 | flags | int(0-255) | Special flags |
| 0x13 | tile_acceptance_list | n*(BB) | Tile acceptance list |



# Industries (INDUSTRY, 0x0A)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Industries

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | substitute_type | int(0-255) | Substitute industry type |
| 0x09 | override_type | int(0-255) | Industry type override |
| 0x0A | layouts | Layouts | Set industry layout(s) |
| 0x0B | production_flags | int(0-255) | Industry production flags |
| 0x0C | closure_message | int(0-65535) | Industry closure message |
| 0x0D | production_increase_message | int(0-65535) | Production increase message |
| 0x0E | production_decrease_message | int(0-65535) | Production decrease message |
| 0x0F | func_cost | int(0-255) | Fund cost multiplier |
| 0x10 | production_cargo | int(0-65535) | Production cargo types |
| 0x11 | acceptance_cargo | int(0-4294967295) | Acceptance cargo types |
| 0x12 | produciton_multipliers | int(0-255) | Production multipliers |
| 0x13 | acceptance_multipliers | int(0-255) | Production multipliers |
| 0x14 | minimal_distributed | int(0-255) | Minimal amount of cargo distributed |
| 0x15 | random_sound | n*B | Random sound effects |
| 0x16 | conflicting_indtypes | 3*B | Conflicting industry types |
| 0x17 | mapgen_probability | int(0-255) | Probability in random game |
| 0x18 | ingame_probability | int(0-255) | Probability during gameplay |
| 0x19 | map_colour | int(0-255) | Map color |
| 0x1A | special_flags | int(0-4294967295) | Special industry flags to define special behavior |
| 0x1B | text_id | int(0-65535) | New industry text ID |
| 0x1C | input_mult1 | int(0-4294967295) | Input cargo multipliers for the three input cargo types |
| 0x1D | input_mult2 | int(0-4294967295) | Input cargo multipliers for the three input cargo types |
| 0x1E | input_mult3 | int(0-4294967295) | Input cargo multipliers for the three input cargo types |
| 0x1F | name | int(0-65535) | Industry name |
| 0x20 | prospecting_chance | int(0-4294967295) | Prospecting success chance |
| 0x21 | cb_flags1 | int(0-255) | Callback flags |
| 0x22 | cb_flags2 | int(0-255) | Callback flags |
| 0x23 | destruction_cost | int(0-4294967295) | Destruction cost multiplier |
| 0x24 | station_text | int(0-65535) | Default text for nearby station |
| 0x25 | production_types | n*B | Production cargo type list |
| 0x26 | acceptance_types | n*B | Acceptance cargo type list |
| 0x27 | produciton_multipliers | n*B | Production multiplier list |
| 0x28 | input_multipliers | n*m*W | Input cargo multiplier list |


# Cargo types (CARGO, 0x0B)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Cargos

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | bit_number | int(0-255) | Bit number for bitmasks |
| 0x09 | type_text | int(0-65535) | TextID for the cargo type name |
| 0x0A | unit_text | int(0-65535) | TextID for the name of one unit from the cargo type |
| 0x0B | one_text | int(0-65535) | TextID to be displayed for 1 unit of cargo |
| 0x0C | many_text | int(0-65535) | TextID to be displayed for multiple units of cargo |
| 0x0D | abbr_text | int(0-65535) | TextID for cargo type abbreviation |
| 0x0E | icon_sprite | int(0-65535) | Sprite number for the icon of the cargo |
| 0x0F | weight | int(0-255) | Weight of one unit of the cargo |
| 0x10 | penalty1 | int(0-255) | Penalty times |
| 0x11 | penalty2 | int(0-255) | Penalty times |
| 0x12 | base_price | int(0-4294967295) | Base price |
| 0x13 | station_colour | int(0-255) | Color for the station list window |
| 0x14 | colour | int(0-255) | Color for the cargo payment list window |
| 0x15 | is_freight | int(0-255) | Freight status (for freight-weight-multiplier setting); 0=not freight, 1=is freight |
| 0x16 | classes | int(0-65535) | Cargo classes |
| 0x17 | label | int(0-4294967295) | Cargo label |
| 0x18 | town_growth_sub | int(0-255) | Substitute type for town growth |
| 0x19 | town_growth_mult | int(0-65535) | Multiplier for town growth |
| 0x1A | cb_flags | int(0-255) | Callback flags |
| 0x1B | units_text | int(0-65535) | TextID for displaying the units of a cargo |
| 0x1C | amount_text | int(0-65535) | TextID for displaying the amount of cargo |
| 0x1D | capacity_mult | int(0-65535) | Capacity mulitplier |


# Sound effects (SOUND_EFFECT, 0x0C)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Sound_Effects

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | relative_volume | int(0-255) | Relative volume |
| 0x09 | priority | int(0-255) | Priority |
| 0x0A | override | int(0-255) | Override old sound |

# Airports (AIRPORT, 0x0D)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Airports

Not supported


# Signals (SIGNAL, 0x0E)

Not supported


# Objects (OBJECT, 0x0F)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Objects

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | class | Label | Class label |
| 0x09 | class_name_id | int(0-65535) | Text ID for class |
| 0x0A | name_id | int(0-65535) | Text ID for this object |
| 0x0B | climates_available | ClimateProperty | Climate availability |
| 0x0C | size | SizeTupleProperty | Byte representing size |
| 0x0D | build_cost_multiplier | int(0-255) | Object build cost factor (sets object removal cost factor as well) |
| 0x0E | introduction_date | DateProperty | Introduction date |
| 0x0F | end_of_life_date | int(0-4294967295) | End of life date |
| 0x10 | flags | int(0-65535) | Object flags |
| 0x11 | animation_info | int(0-65535) | Animation information |
| 0x12 | animation_speed | int(0-255) | Animation speed |
| 0x13 | animation_triggers | int(0-65535) | Animation triggers |
| 0x14 | remove_cost_multiplier | int(0-255) | Object removal cost factor (set after object build cost factor) |
| 0x15 | cb_flags | int(0-65535) | Callback flags, see below |
| 0x16 | height | int(0-255) | Height of the building |
| 0x17 | num_views | int(0-255) | Number of object views |
| 0x18 | count_per_map256 | int(0-255) | Measure for number of objects placed upon map creation |


# Railtypes (RAILTYPE, 0x10)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Railtypes

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | label | Label | Rail type label |
| 0x09 | toolbar_caption | int(0-65535) | StringID: Build rail toolbar caption |
| 0x0A | menu_text | int(0-65535) | StringID: Rail construction dropdown text |
| 0x0B | build_window_caption | int(0-65535) | StringID: Build vehicle window caption |
| 0x0C | autoreplace_text | int(0-65535) | StringID: Autoreplace text |
| 0x0D | new_engine_text | int(0-65535) | StringID: New engine text |
| 0x13 | construction_cost | int(0-65535) | Construction costs |
| 0x16 | map_colour | int(0-255) | Minimap colour |
| 0x17 | introduction_date | DateProperty | Introduction date |
| 0x1A | sort_order | int(0-255) | Sort order |
| 0x1B | name | int(0-65535) | StringID: Rail type name |
| 0x1C | maintenance_cost | int(0-65535) | Infrastructure maintenance cost factor |
| 0x0E | compatible_railtype_list | list[Label] | Compatible rail type list |
| 0x0F | powered_railtype_list | list[Label] | Powered rail type list |
| 0x10 | railtype_flags | int(0-255) | Rail type flags |
| 0x11 | curve_speed_multiplier | int(0-255) | Curve speed advantage multiplier |
| 0x12 | station_graphics | int(0-255) | Station (and depot) graphics |
| 0x14 | speed_limit | int(0-65535) | Speed limit |
| 0x15 | acceleration_model | int(0-255) | Acceleration model |
| 0x18 | requires_railtype_list | list[Label] | Introduction required rail type list |
| 0x19 | introduces_railtype_list | list[Label] | Introduced rail type list |
| 0x1D | alternative_railtype_list | list[Label] | Alternate rail type labels that shall be "redirected" to this rail type |


# Airport tiles (AIRPORT_TILE, 0x11)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Airports

Not supported


# Road types (ROADTYPE, 0x12)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Roadtypes

| NFO# | Name | Type | Description |
| - | - | - | - |
| 0x08 | label | Label | Road type label |
| 0x09 | toolbar_caption | int(0-65535) | StringID: Build road toolbar caption |
| 0x0A | menu_text | int(0-65535) | StringID: Road construction dropdown text |
| 0x0B | build_window_caption | int(0-65535) | StringID: Build vehicle window caption |
| 0x0C | autoreplace_text | int(0-65535) | StringID: Autoreplace text |
| 0x0D | new_engine_text | int(0-65535) | StringID: New engine text |
| 0x0F | powered_roadtype_list | list[Label] | Powered road type list |
| 0x10 | roadtype_flags | int(0-255) | Road type flags |
| 0x13 | construction_cost | int(0-65535) | Construction costs |
| 0x14 | speed_limit | int(0-65535) | Speed limit |
| 0x16 | map_colour | int(0-255) | Minimap colour |
| 0x17 | introduction_date | DateProperty | Introduction date |
| 0x18 | requires_roadtype_list | list[Label] | Introduction required road type list |
| 0x19 | introduces_roadtype_list | list[Label] | Introduced road type list |
| 0x1A | sort_order | int(0-255) | Sort order |
| 0x1B | name | int(0-65535) | StringID: Road type name |
| 0x1C | maintenance_cost | int(0-65535) | Infrastructure maintenance cost factor |
| 0x1D | alternative_roadtype_list | list[Label] | Alternate road type labels that shall be "redirected" to this road type |


# Tram types (TRAMTYPE, 0x13)
https://newgrf-specs.tt-wiki.net/wiki/Action0/Tramtypes

 Not supported


# Road stops

 Not supported
