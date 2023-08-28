from pathlib import Path

import grf

from .common import check_action

SCRIPT_DIR = Path(__file__).resolve().parent

def test_string_ref():
	s = grf.StringManager()
	s.import_lang_dir(str(SCRIPT_DIR / 'lang'))
	check_action(
        grf.SetProperties(
			{'INFO': {'NAME': s['NAME']}}
		),
		'14:43:49:4e:46:4f:54:4e:41:4d:45:7f:6e:61:6d:65:5f:67:62:00:54:4e:41:4d:45:00:6e:61:6d:65:5f:75:73:00:00:00',
	)
