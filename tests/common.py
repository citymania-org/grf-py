import tempfile
import os

from nose.tools import eq_

from grf import BaseNewGRF

def _do_check_lib(newgrf, obj, result):
	newgrf.add(obj)
	sprites = newgrf.generate_sprites()
	sprites = newgrf.resolve_refs(sprites)
	sprites.extend(newgrf.strings.get_actions())

	for s in sprites:
		print(s.py(None))
	eq_(len(sprites), len(result))
	for i, s, r in zip(range(len(sprites)), sprites, result):
		if s != r:
			print(f'Difference at position {i}')
			print('Result:')
			print(s.py(None))
			print('Expected:')
			print(r.py(None))
			assert False


def check_lib(prepare, obj, result):
	with tempfile.TemporaryDirectory() as tmp:
		index_path = os.path.join(tmp, 'id_index.json')
		with open(index_path, 'w') as f:
			f.write('{"version": 1, "index": {}}')
		newgrf = BaseNewGRF(id_index_path=index_path)
		if prepare is not None:
			prepare(newgrf)
		_do_check_lib(newgrf, obj, result)
