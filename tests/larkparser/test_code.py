from grf.larkparser import Code

from grf.actions import If, Label
from ..common import check_lib


def test_string_ref():
	check_lib(
	    lambda g: g.register_parameter('testparam'),
	    Code.If('PARAM.testparam == 35', [Label(55, 'if body')]),
	    [
	        If(
	        	is_static=False,
	        	variable=0,
	        	condition=2,
	        	value=35,
	            skip=1,
	            varsize=4,
	        ),
	        Label(55, 'if body'),
	        Label(1),
	    ]
	)
