import argparse

from .grf import BaseNewGRF


def build_func(g, grf_file, args):
	print(f'Building {grf_file}')
	g.write(grf_file)


def main(g, grf_file):
	assert isinstance(g, BaseNewGRF)
	assert isinstance(grf_file, str)
	parser = argparse.ArgumentParser(description=g.name)

	# Create subparsers for different commands
	subparsers = parser.add_subparsers(title='commands', dest='command')

	# Create a parser for the 'build' command
	build_parser = subparsers.add_parser('build', help='Build newgrf')
	# create_parser.add_argument('--name', required=True, help='Name of the item')
	# create_parser.add_argument('--size', type=int, required=True, help='Size of the item')
	build_parser.set_defaults(func=lambda args: build_func(g, grf_file, args))

	args = parser.parse_args()

	if args.command is None:
		build_func(g, grf_file, None)
	else:
		args.func(args)

