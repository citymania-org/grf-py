import grf

from .common import check_action


def test_basic_format():
	check_action(
		grf.Action1(
			feature=grf.TRAIN,
			set_count=255,
			sprite_count=0x5E07,

		),
		'01:00:ff:ff:07:5e',
	)

	check_action(
		grf.SpriteSet(
			feature=grf.TRAIN,
			count=0x5E07,

		),
		'01:00:01:ff:07:5e',
	)


def test_extended_format():
	check_action(
		grf.Action1(
			feature=grf.TRAIN,
			set_count=256,
			sprite_count=0x5E07,
		),
		'01:00:00:ff:00:00:ff:00:01:ff:07:5e',
	)

	# first_set=0 forces extended format even though basic could be used
	check_action(
		grf.Action1(
			feature=grf.TRAIN,
			set_count=255,
			sprite_count=0x5E07,
			first_set=0,
		),
		'01:00:00:ff:00:00:ff:ff:00:ff:07:5e',
	)

	check_action(
		grf.Action1(
			feature=grf.TRAIN,
			set_count=0x57C0,
			sprite_count=0x5E07,
			first_set=0x15E7,
		),
		'01:00:00:ff:e7:15:ff:c0:57:ff:07:5e',
	)


