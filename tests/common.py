import tempfile
import os

import pytest

from grf import BaseNewGRF, hex_str, WriteContext


def _do_check_grf(newgrf, expected):
	result = newgrf.generate_sprites()
	result = newgrf.resolve_refs(result)
	result.extend(newgrf.strings.get_actions())

	def print_sprites():
		print('Generated output: ')
		for s in result:
			print(s.py(None).rstrip(), end=',')
		print()

	for i, s, r in zip(range(len(result)), result, expected):
		if s != r:
			print_sprites()
			print(f'Difference at position {i}')
			print('Result:')
			print(s.py(None))
			print('Expected:')
			print(r.py(None))
			assert False

	if len(result) != len(expected):
		print_sprites()
		if len(result) > len(expected):
			print(f'Result has more actions than expected:')
			for s in result[len(expected):]:
				print(s.py(None))
		else:
			print(f'Expected has more actions than result:')
			for s in expected[len(result):]:
				print(s.py(None))
		assert False


def check_grf(grf_factory, result):
	with tempfile.TemporaryDirectory() as tmp:
		map_file = os.path.join(tmp, 'id_index.json')
		with open(map_file, 'w') as f:
			f.write('{"version": 1, "index": {}}')
		newgrf = grf_factory(id_map_file=map_file)
		_do_check_grf(newgrf, result)


def check_lib(prepare, obj, result):
	def func(id_map_file):
		g = BaseNewGRF(id_map_file=id_map_file)
		if prepare is not None:
			prepare(g)
		g.add(obj)
		return g
	check_grf(func, result)


def check_action(action, result):
	result = ''.join(result.split()).lower()
	encoded = action.get_data(WriteContext())
	encoded_hex = hex_str(encoded)
	assert encoded_hex == result


# pytest equivalent of nose @raises
def raises(*exceptions):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with pytest.raises(exceptions):
                fn(*args, **kwargs)
        return wrapper
    return decorator
