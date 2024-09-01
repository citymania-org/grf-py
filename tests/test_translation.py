from pathlib import Path

from nose.tools import eq_

import grf

from .common import check_grf


SCRIPT_DIR = Path(__file__).resolve().parent


def check_lib(prepare, obj, result):
	with tempfile.TemporaryDirectory() as tmp:
		map_file = os.path.join(tmp, 'id_index.json')
		with open(map_file, 'w') as f:
			f.write('{"version": 1, "index": {}}')
		newgrf = BaseNewGRF(id_map_file=map_file)
		if prepare is not None:
			prepare(newgrf)
		newgrf.add(obj)
		_do_check_lib(newgrf, result)


def test_newgrf_name_and_descripription_translation():
	s = grf.StringManager()
	s.import_lang_dir(str(SCRIPT_DIR / 'lang'))
	def func(id_map_file):
		g = grf.NewGRF(
			grfid='GPTT',
			name=s['NAME'],
			description=s['DESCRIPTION'],
			id_map_file=id_map_file,
			strings=s,
		)
		return g
	check_grf(func, [
		grf.SetProperties({
		    'INFO': {
		    	'PALS': b'D',
		    	'NAME': s['NAME'],
		    	'DESC': s['DESCRIPTION'],
		    }
		}),
        grf.SetDescription(
            format_version=8,
            grfid=b'GPTT',
            name=b'name_\x8bgb',
            description=b'description_\x8bgb',
        ),
        # grf.SetProperties(
		# 	{'INFO': {'NAME': g.strings['NAME']}}
		# ),
        grf.Define(
            feature=grf.GLOBAL_VAR,
            id=0,
            props={'lang_plural': 0}
        ),
        grf.Define(
            feature=grf.GLOBAL_VAR,
            id=1,
            props={'lang_plural': 0}
        ),
	])
