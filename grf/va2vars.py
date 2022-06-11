from .common import TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, CANAL, BRIDGE, HOUSE, INDUSTRY_TILE, INDUSTRY, \
                    CARGO, SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, AIRPORT_TILE, ROADTYPE, TRAMTYPE


NML_VA2_GLOBALVARS = {
    'current_month'        : {'var': 0x02, 'start':  0, 'size':  8},
    'current_day_of_month' : {'var': 0x02, 'start':  8, 'size':  5},
    'is_leapyear'          : {'var': 0x02, 'start': 15, 'size':  1},
    'current_day_of_year'  : {'var': 0x02, 'start': 16, 'size':  9},
    'traffic_side'         : {'var': 0x06, 'start':  4, 'size':  1},
    'animation_counter'    : {'var': 0x0A, 'start':  0, 'size': 16},
    'current_callback_byte': {'var': 0x0C, 'start':  0, 'size': 8},
    'current_callback'     : {'var': 0x0C, 'start':  0, 'size': 16},
    'extra_callback_info1' : {'var': 0x10, 'start':  0, 'size': 32},
    'extra_callback_info1_byte' : {'var': 0x10, 'start':  0, 'size': 8},
    'extra_callback_info1_byte_2' : {'var': 0x10, 'start':  8, 'size': 8},
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
value_sign_extend = None
value_mul_div = lambda x, y: None
value_add_constant = lambda x: None
value_equals = lambda x: None

NML_VA2_VEHICLES = {
    'grfid'                            : {'var': 0x25, 'start':  0, 'size': 32},
    'position_in_consist'              : {'var': 0x40, 'start':  0, 'size':  8},
    'position_in_consist_from_end'     : {'var': 0x40, 'start':  8, 'size':  8},
    'num_vehs_in_consist'              : {'var': 0x40, 'start': 16, 'size':  8, 'value_function': value_add_constant(1)}, # Zero-based, add 1 to make sane
    'position_in_vehid_chain'          : {'var': 0x41, 'start':  0, 'size':  8},
    'position_in_vehid_chain_from_end' : {'var': 0x41, 'start':  8, 'size':  8},
    'num_vehs_in_vehid_chain'          : {'var': 0x41, 'start': 16, 'size':  8}, # One-based, already sane
    'cargo_classes_in_consist'         : {'var': 0x42, 'start':  0, 'size':  8},
    'most_common_cargo_type'           : {'var': 0x42, 'start':  8, 'size':  8},
    'most_common_cargo_subtype'        : {'var': 0x42, 'start': 16, 'size':  8},
    'bitmask_consist_info'             : {'var': 0x42, 'start': 24, 'size':  8},
    'company_num'                      : {'var': 0x43, 'start':  0, 'size':  8},
    'company_type'                     : {'var': 0x43, 'start': 16, 'size':  2},
    'company_colour1'                  : {'var': 0x43, 'start': 24, 'size':  4},
    'company_colour2'                  : {'var': 0x43, 'start': 28, 'size':  4},
    'aircraft_height'                  : {'var': 0x44, 'start':  8, 'size':  8},
    'airport_type'                     : {'var': 0x44, 'start':  0, 'size':  8},
    'curv_info_prev_cur'               : {'var': 0x45, 'start':  0, 'size':  4, 'value_function': value_sign_extend},
    'curv_info_cur_next'               : {'var': 0x45, 'start':  8, 'size':  4, 'value_function': value_sign_extend},
    'curv_info_prev_next'              : {'var': 0x45, 'start': 16, 'size':  4, 'value_function': value_sign_extend},
    'curv_info'                        : {'var': 0x45, 'start':  0, 'size': 12},
            # 'value_function': lambda var, info: nmlop.AND(var, 0x0F0F).reduce()},
    'motion_counter'                   : {'var': 0x46, 'start':  8, 'size': 24},
    'cargo_type_in_veh'                : {'var': 0x47, 'start':  0, 'size':  8},
    'cargo_unit_weight'                : {'var': 0x47, 'start':  8, 'size':  8},
    'cargo_classes'                    : {'var': 0x47, 'start': 16, 'size': 16},
    'vehicle_is_available'             : {'var': 0x48, 'start':  0, 'size':  1},
    'vehicle_is_testing'               : {'var': 0x48, 'start':  1, 'size':  1},
    'vehicle_is_offered'               : {'var': 0x48, 'start':  2, 'size':  1},
    'build_year'                       : {'var': 0x49, 'start':  0, 'size': 32},
    'vehicle_is_potentially_powered'   : {'var': 0x4A, 'start':  8, 'size':  1},
    'tile_has_catenary'                : {'var': 0x4A, 'start':  9, 'size':  1},
    'date_of_last_service'             : {'var': 0x4B, 'start':  0, 'size': 32},
    'position_in_articulated_veh'          : {'var': 0x4D, 'start':  0, 'size':  8},
    'position_in_articulated_veh_from_end' : {'var': 0x4D, 'start':  8, 'size':  8},
    'waiting_triggers'                 : {'var': 0x5F, 'start':  0, 'size':  8},
    'random_bits'                      : {'var': 0x5F, 'start':  8, 'size':  8},
    'direction'                        : {'var': 0x9F, 'start':  0, 'size':  8},
    'vehicle_is_hidden'                : {'var': 0xB2, 'start':  0, 'size':  1},
    'vehicle_is_stopped'               : {'var': 0xB2, 'start':  1, 'size':  1},
    'vehicle_is_crashed'               : {'var': 0xB2, 'start':  7, 'size':  1},
    'cargo_capacity'                   : {'var': 0xBA, 'start':  0, 'size': 16},
    'cargo_count'                      : {'var': 0xBC, 'start':  0, 'size': 16},
    'age_in_days'                      : {'var': 0xC0, 'start':  0, 'size': 16},
    'max_age_in_days'                  : {'var': 0xC2, 'start':  0, 'size': 16},
    'vehicle_type_id'                  : {'var': 0xC6, 'start':  0, 'size': 16},
    'vehicle_is_flipped'               : {'var': 0xC8, 'start':  1, 'size':  1},
    'breakdowns_since_last_service'    : {'var': 0xCA, 'start':  0, 'size':  8},
    'vehicle_is_broken'                : {'var': 0xCB, 'start':  0, 'size':  8, 'value_function': value_equals(1)},
    'reliability'                      : {'var': 0xCE, 'start':  0, 'size': 16, 'value_function': value_mul_div(101, 0x10000)},
    'cargo_subtype'                    : {'var': 0xF2, 'start':  0, 'size':  8},
    'vehicle_is_unloading'             : {'var': 0xFE, 'start':  1, 'size':  1},
    'vehicle_is_powered'               : {'var': 0xFE, 'start':  5, 'size':  1},
    'vehicle_is_not_powered'           : {'var': 0xFE, 'start':  6, 'size':  1},
    'vehicle_is_reversed'              : {'var': 0xFE, 'start':  8, 'size':  1},
    'built_during_preview'             : {'var': 0xFE, 'start': 10, 'size':  1},
}

NML_VA2_TRAIN = {
    **NML_VA2_VEHICLES,
    #0x4786 / 0x10000 is an approximation of 3.5790976, the conversion factor
    #for train speed
    'max_speed'           : {'var': 0x98, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x4786, 0x10000)},
    'current_speed'       : {'var': 0xB4, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x4786, 0x10000)},
    'current_railtype'    : {'var': 0x4A, 'start':  0, 'size':  8},
    'current_max_speed'   : {'var': 0x4C, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x4786, 0x10000)},
    'vehicle_is_in_depot' : {'var': 0xE2, 'start': 7, 'size':  1},
}

NML_VA2_RV = {
    **NML_VA2_VEHICLES,
    #0x23C3 / 0x10000 is an approximation of 7.1581952, the conversion factor
    #for road vehicle speed
    'max_speed'           : {'var': 0x98, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x23C3, 0x10000)},
    'current_speed'       : {'var': 0xB4, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x23C3, 0x10000)},
    'current_roadtype'    : {'var': 0x4A, 'start':  0, 'size':  8},
    'current_tramtype'    : {'var': 0x4A, 'start':  0, 'size':  8},
    'current_max_speed'   : {'var': 0x4C, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x23C3, 0x10000)},
    'vehicle_is_in_depot' : {'var': 0xE2, 'start': 0, 'size':  8, 'value_function': value_equals(0xFE)},
}

NML_VA2_SHIP = {
    **NML_VA2_VEHICLES,
    #0x23C3 / 0x10000 is an approximation of 7.1581952, the conversion factor
    #for ship speed
    'max_speed'           : {'var': 0x98, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x23C3, 0x10000)},
    'current_speed'       : {'var': 0xB4, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x23C3, 0x10000)},
    'current_max_speed'   : {'var': 0x4C, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x23C3, 0x10000)},
    'vehicle_is_in_depot' : {'var': 0xE2, 'start': 7, 'size':  1},
}

NML_VA2_AIRCRAFT = {
    **NML_VA2_VEHICLES,
    #0x3939 / 0x1000 is an approximation of 0.279617, the conversion factor
    #Note that the denominator has one less zero here!
    #for aircraft speed
    'max_speed'           : {'var': 0x98, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x3939, 0x1000)},
    'current_speed'       : {'var': 0xB4, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x3939, 0x1000)},
    'current_max_speed'   : {'var': 0x4C, 'start': 0, 'size': 16, 'value_function': value_mul_div(0x3939, 0x1000)},
    'vehicle_is_in_depot' : {'var': 0xE6, 'start': 0, 'size':  8, 'value_function': value_equals(0)},
}


NML_VA2_BASE_STATIONS = {
    # Var 48 doesn't work with newcargos, do not use
    'had_vehicle_of_type' : {'var': 0x8A, 'start': 1, 'size': 5}, # Only read bits 1-5
    'is_waypoint'         : {'var': 0x8A, 'start': 6, 'size': 1},
    'facilities'          : {'var': 0xF0, 'start': 0, 'size': 8},
    'airport_type'        : {'var': 0xF1, 'start': 0, 'size': 8},
    # Variables F2, F3, F6 (roadstop, airport flags) are next to useless
    # Also, their values are not the same as in TTDP / spec
    # Therefore, these are not implemented
    'build_date'          : {'var': 0xFA, 'start': 0, 'size': 16, 'value_function': value_add_constant(701265)},
    'cargo_amount_waiting'      : {'var': 0x60, 'start': 0, 'size': 32},
    'cargo_time_since_pickup'   : {'var': 0x61, 'start': 0, 'size': 32},
    'cargo_rating'              : {'var': 0x62, 'start': 0, 'size': 32, 'value_function': value_mul_div(101, 256)},
    'cargo_time_en_route'       : {'var': 0x63, 'start': 0, 'size': 32},
    'cargo_last_vehicle_speed'  : {'var': 0x64, 'start': 0, 'size':  8},
    'cargo_last_vehicle_age'    : {'var': 0x64, 'start': 8, 'size':  8},
    'cargo_accepted'            : {'var': 0x65, 'start': 3, 'size':  1},
    'cargo_accepted_ever'       : {'var': 0x69, 'start': 0, 'size':  1},
    'cargo_accepted_last_month' : {'var': 0x69, 'start': 1, 'size':  1},
    'cargo_accepted_this_month' : {'var': 0x69, 'start': 2, 'size':  1},
    'cargo_accepted_bigtick'    : {'var': 0x69, 'start': 3, 'size':  1},
}


NML_VA2_STATIONS = {
    **NML_VA2_BASE_STATIONS,
    # Vars 40, 41, 46, 47, 49 are implemented as 60+x vars,
    # except for the 'tile type' part which is always the same anyways
    'tile_type'                : {'var': 0x40, 'start': 24, 'size': 4},
    'terrain_type'             : {'var': 0x42, 'start':  0, 'size': 8},
    'track_type'               : {'var': 0x42, 'start':  8, 'size': 8},
    'company_num'              : {'var': 0x43, 'start':  0, 'size': 8},
    'company_type'             : {'var': 0x43, 'start': 16, 'size': 2},
    'company_colour1'          : {'var': 0x43, 'start': 24, 'size': 4},
    'company_colour2'          : {'var': 0x43, 'start': 28, 'size': 4},
    'pbs_reserved'             : {'var': 0x44, 'start':  0, 'size': 1},
    'pbs_reserved_or_disabled' : {'var': 0x44, 'start':  1, 'size': 1},
    'pbs_enabled'              : {'var': 0x44, 'start':  2, 'size': 1},
    'rail_continuation'        : {'var': 0x45, 'start':  0, 'size': 8},
    'rail_present'             : {'var': 0x45, 'start':  8, 'size': 8},
    'animation_frame'          : {'var': 0x4A, 'start':  0, 'size': 8},
}


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

NML_VA2_INDUSTRY_TILES = {
    'construction_state' : {'var': 0x40, 'start': 0, 'size':  2},
    'terrain_type'       : {'var': 0x41, 'start': 0, 'size':  8},
    'town_zone'          : {'var': 0x42, 'start': 0, 'size':  3},
    'relative_x'         : {'var': 0x43, 'start': 0, 'size':  8},
    'relative_y'         : {'var': 0x43, 'start': 8, 'size':  8},
    'relative_pos'       : {'var': 0x43, 'start': 0, 'size': 16},
    'animation_frame'    : {'var': 0x44, 'start': 0, 'size':  8},
    'random_bits'        : {'var': 0x5F, 'start': 8, 'size':  8},
    'nearby_tile_slope'            : {'var': 0x60, 'start':  0, 'size':  5, 'param_function': signed_tile_offset},
    'nearby_tile_is_same_industry' : {'var': 0x60, 'start':  8, 'size':  1, 'param_function': signed_tile_offset},
    'nearby_tile_is_water'         : {'var': 0x60, 'start':  9, 'size':  1, 'param_function': signed_tile_offset},
    'nearby_tile_terrain_type'     : {'var': 0x60, 'start': 10, 'size':  3, 'param_function': signed_tile_offset},
    'nearby_tile_water_class'      : {'var': 0x60, 'start': 13, 'size':  2, 'param_function': signed_tile_offset},
    'nearby_tile_height'           : {'var': 0x60, 'start': 16, 'size':  8, 'param_function': signed_tile_offset},
    'nearby_tile_class'            : {'var': 0x60, 'start': 24, 'size':  4, 'param_function': signed_tile_offset},
    'nearby_tile_animation_frame'  : {'var': 0x61, 'start':  0, 'size':  8, 'param_function': signed_tile_offset},
    'nearby_tile_industrytile_id'  : {'var': 0x62, 'start':  0, 'size': 16, 'param_function': signed_tile_offset},
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

NML_VA2_RAILTYPES = {
    'terrain_type'          : {'var': 0x40, 'start': 0, 'size':  8},
    'enhanced_tunnels'      : {'var': 0x41, 'start': 0, 'size':  8},
    'level_crossing_status' : {'var': 0x42, 'start': 0, 'size':  8},
    'build_date'            : {'var': 0x43, 'start': 0, 'size': 32},
    'town_zone'             : {'var': 0x44, 'start': 0, 'size':  8},
    'random_bits'           : {'var': 0x5F, 'start': 8, 'size':  2},
}
# Railtypes have no 60+x variables



VA2_VARS = {
    feature: {**NML_VA2_GLOBALVARS, **d}
    for feature, d in (
        (INDUSTRY_TILE, NML_VA2_INDUSTRY_TILES),
        (TRAIN, NML_VA2_TRAIN),
        (RV, NML_VA2_RV),
        (SHIP, NML_VA2_SHIP),
        (AIRCRAFT, NML_VA2_AIRCRAFT),
        (STATION, NML_VA2_STATIONS),
        (INDUSTRY, NML_VA2_INDUSTRIES),
        (OBJECT, NML_VA2_OBJECTS),
        (RAILTYPE, NML_VA2_RAILTYPES),
    )
}


VA2_VARS_INV = {
    feature: {(v['var'], v['start'], (1 << v['size']) - 1): k for k, v in d.items()}
    for feature, d in VA2_VARS.items()
}
