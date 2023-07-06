from nose.tools import eq_

from grf import BaseNewGRF


def check_lib(obj, result, *, newgrf=None):
	if newgrf is None:
		newgrf = BaseNewGRF()
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
