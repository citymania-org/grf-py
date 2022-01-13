
NML_VA2_GLOBALVARS = {
    'current_month'        : {'var': 0x02, 'start':  0, 'size':  8},
    'current_day_of_month' : {'var': 0x02, 'start':  8, 'size':  5},
    'is_leapyear'          : {'var': 0x02, 'start': 15, 'size':  1},
    'current_day_of_year'  : {'var': 0x02, 'start': 16, 'size':  9},
    'traffic_side'         : {'var': 0x06, 'start':  4, 'size':  1},
    'animation_counter'    : {'var': 0x0A, 'start':  0, 'size': 16},
    'current_callback'     : {'var': 0x0C, 'start':  0, 'size': 16},
    'extra_callback_info1' : {'var': 0x10, 'start':  0, 'size': 32},
    'game_mode'            : {'var': 0x12, 'start':  0, 'size':  8},
    'extra_callback_info2' : {'var': 0x18, 'start':  0, 'size': 32},
    'display_options'      : {'var': 0x1B, 'start':  0, 'size':  6},
    'last_computed_result' : {'var': 0x1C, 'start':  0, 'size': 32},
    'snowline_height'      : {'var': 0x20, 'start':  0, 'size':  8},
    'difficulty_level'     : {'var': 0x22, 'start':  0, 'size':  8},
    'current_date'         : {'var': 0x23, 'start':  0, 'size': 32},
    'current_year'         : {'var': 0x24, 'start':  0, 'size': 32},
}


industry_count = signed_tile_offset = None
value_mul_div = lambda x, y: None
value_add_constant = lambda x: None

NML_VA2_OBJECTS = {
    'relative_x'             : {'var': 0x40, 'start':  0, 'size':  8},
    'relative_y'             : {'var': 0x40, 'start':  8, 'size':  8},
    'relative_pos'           : {'var': 0x40, 'start':  0, 'size': 16},

    'terrain_type'           : {'var': 0x41, 'start':  0, 'size':  3},
    'tile_slope'             : {'var': 0x41, 'start':  8, 'size':  5},

    'build_date'             : {'var': 0x42, 'start':  0, 'size': 32},

    'animation_frame'        : {'var': 0x43, 'start':  0, 'size':  8},
    'company_colour'         : {'var': 0x43, 'start':  0, 'size':  8},

    'owner'                  : {'var': 0x44, 'start':  0, 'size':  8},

    'town_manhattan_dist'    : {'var': 0x45, 'start':  0, 'size': 16},
    'town_zone'              : {'var': 0x45, 'start': 16, 'size':  8},

    'town_euclidean_dist'    : {'var': 0x46, 'start':  0, 'size': 16},
    'view'                   : {'var': 0x48, 'start':  0, 'size':  8},
    'random_bits'            : {'var': 0x5F, 'start':  8, 'size':  8},

    'tile_height'            : {'var': 0x62, 'start':  16, 'size':  8, 'param': 0},

    'nearby_tile_object_type'      : {'var': 0x60, 'start':  0, 'size': 16, 'param_function': signed_tile_offset},
    'nearby_tile_object_view'      : {'var': 0x60, 'start': 16, 'size':  4, 'param_function': signed_tile_offset},

    'nearby_tile_random_bits'      : {'var': 0x61, 'start':  0, 'size':  8, 'param_function': signed_tile_offset},

    'nearby_tile_slope'            : {'var': 0x62, 'start':  0, 'size':  5, 'param_function': signed_tile_offset},
    'nearby_tile_is_same_object'   : {'var': 0x62, 'start':  8, 'size':  1, 'param_function': signed_tile_offset},
    'nearby_tile_is_water'         : {'var': 0x62, 'start':  9, 'size':  1, 'param_function': signed_tile_offset},
    'nearby_tile_terrain_type'     : {'var': 0x62, 'start': 10, 'size':  3, 'param_function': signed_tile_offset},
    'nearby_tile_water_class'      : {'var': 0x62, 'start': 13, 'size':  2, 'param_function': signed_tile_offset},
    'nearby_tile_height'           : {'var': 0x62, 'start': 16, 'size':  8, 'param_function': signed_tile_offset},
    'nearby_tile_class'            : {'var': 0x62, 'start': 24, 'size':  4, 'param_function': signed_tile_offset},

    'nearby_tile_animation_frame'  : {'var': 0x63, 'start':  0, 'size':  8, 'param_function': signed_tile_offset},

    'object_count'                 : {'var': 0x64, 'start': 16, 'size':  8, 'param_function': industry_count},
    'object_distance'              : {'var': 0x64, 'start':  0, 'size': 16, 'param_function': industry_count},
}

NML_VA2_INDUSTRIES = {
    'waiting_cargo_1'              : {'var': 0x40, 'start':  0, 'size': 16, 'replaced_by': 'incoming_cargo_waiting'},
    'waiting_cargo_2'              : {'var': 0x41, 'start':  0, 'size': 16, 'replaced_by': 'incoming_cargo_waiting'},
    'waiting_cargo_3'              : {'var': 0x42, 'start':  0, 'size': 16, 'replaced_by': 'incoming_cargo_waiting'},
    'water_distance'               : {'var': 0x43, 'start':  0, 'size': 32},
    'layout_num'                   : {'var': 0x44, 'start':  0, 'size':  8},
    # bits 0 .. 16 are either useless or already covered by var A7
    'founder_type'                 : {'var': 0x45, 'start': 16, 'size':  2},
    'founder_colour1'              : {'var': 0x45, 'start': 24, 'size':  4},
    'founder_colour2'              : {'var': 0x45, 'start': 28, 'size':  4},
    'build_date'                   : {'var': 0x46, 'start':  0, 'size': 32},
    'random_bits'                  : {'var': 0x5F, 'start':  8, 'size': 16},
    'produced_cargo_waiting_1'     : {'var': 0x8A, 'start':  0, 'size': 16, 'replaced_by': 'produced_cargo_waiting'},
    'produced_cargo_waiting_2'     : {'var': 0x8C, 'start':  0, 'size': 16, 'replaced_by': 'produced_cargo_waiting'},
    'production_rate_1'            : {'var': 0x8E, 'start':  0, 'size':  8, 'replaced_by': 'production_rate'},
    'production_rate_2'            : {'var': 0x8F, 'start':  0, 'size':  8, 'replaced_by': 'production_rate'},
    'production_level'             : {'var': 0x93, 'start':  0, 'size':  8},
    'produced_this_month_1'        : {'var': 0x94, 'start':  0, 'size': 16, 'replaced_by': 'this_month_production'},
    'produced_this_month_2'        : {'var': 0x96, 'start':  0, 'size': 16, 'replaced_by': 'this_month_production'},
    'transported_this_month_1'     : {'var': 0x98, 'start':  0, 'size': 16, 'replaced_by': 'this_month_transported'},
    'transported_this_month_2'     : {'var': 0x9A, 'start':  0, 'size': 16, 'replaced_by': 'this_month_transported'},
    'transported_last_month_pct_1' : {'var': 0x9C, 'start':  0, 'size':  8, 'value_function': value_mul_div(101, 256), 'replaced_by': 'transported_last_month_pct'},
    'transported_last_month_pct_2' : {'var': 0x9D, 'start':  0, 'size':  8, 'value_function': value_mul_div(101, 256), 'replaced_by': 'transported_last_month_pct'},
    'produced_last_month_1'        : {'var': 0x9E, 'start':  0, 'size': 16, 'replaced_by': 'last_month_production'},
    'produced_last_month_2'        : {'var': 0xA0, 'start':  0, 'size': 16, 'replaced_by': 'last_month_production'},
    'transported_last_month_1'     : {'var': 0xA2, 'start':  0, 'size': 16, 'replaced_by': 'last_month_transported'},
    'transported_last_month_2'     : {'var': 0xA4, 'start':  0, 'size': 16, 'replaced_by': 'last_month_transported'},
    'founder'                      : {'var': 0xA7, 'start':  0, 'size':  8},
    'colour'                       : {'var': 0xA8, 'start':  0, 'size':  8},
    'counter'                      : {'var': 0xAA, 'start':  0, 'size': 16},
    'build_type'                   : {'var': 0xB3, 'start':  0, 'size':  2},
    'last_accept_date'             : {'var': 0xB4, 'start':  0, 'size': 16, 'value_function': value_add_constant(701265)},
}


VA2_VARS = {
    feature: {**NML_VA2_GLOBALVARS, **d}
    for feature, d in ((0x0a, NML_VA2_INDUSTRIES), (0x0f, NML_VA2_OBJECTS))
}


VA2_VARS_INV = {
    feature: {(v['var'], v['start'], (1 << v['size']) - 1): k for k, v in d.items()}
    for feature, d in VA2_VARS.items()
}

