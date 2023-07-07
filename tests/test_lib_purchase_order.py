import types

from nose.tools import eq_

from grf import BaseNewGRF, SetPurchaseOrder, TRAIN, VariantGroup, Ref, DefineStrings, Define, Action3, Switch

from .common import check_lib


class DummyFeature:
    def __init__(self, id):
        self.id = id
        self.feature = TRAIN
        self.callbacks = types.SimpleNamespace()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, f):
        if isinstance(f, int):
            return self.id == f

        return self.id == f.id

    def __repr__(self):
        return repr(self.id)


def string(ofs, s):
    return DefineStrings(
        feature=TRAIN,
        lang=127,
        offset=ofs,
        is_generic_offset=True,
        strings=[s.encode('utf-8')]
    )


def sortvg(id, sort, vg=None):
    vgprops = {} if vg is None else {'variant_group': vg}
    sortprops = {} if sort is None else {'sort_purchase_list': sort}
    return Define(
        feature=TRAIN,
        id=id,
        props={
            **vgprops,
            **sortprops,
        }
    )


def test_lib_train_basics():
    f1 = DummyFeature(1)
    fa1 = DummyFeature(11)
    fa2 = DummyFeature(12)
    fab1 = DummyFeature(111)
    fab2 = DummyFeature(112)
    facd1 = DummyFeature(11111)
    facd2 = DummyFeature(11112)
    face1 = DummyFeature(11121)
    face2 = DummyFeature(11122)
    fac1 = DummyFeature(1111)

    order = SetPurchaseOrder(
        f1,
        VariantGroup(
            'A',
            fa1,
            fa2,
            VariantGroup(
                'AB',
                fab1,
                fab2,
            ),
            VariantGroup(
                'AC',
                fac1,
                VariantGroup(
                    'ACD',
                    facd1,
                    facd2,
                ),
                VariantGroup(
                    'ACE',
                    face1,
                    face2,
                ),
            ),
        )
    )

    check_lib(
        lambda g: order.set_variant_callbacks(g),
        order,
        [
            sortvg(11122, None, 11121),
            sortvg(11121, 11122, 1111),
            sortvg(11112, 11121, 11111),
            sortvg(11111, 11112, 1111),
            sortvg(1111, 11111, 11),
            sortvg(112, 1111, 111),
            sortvg(111, 112, 11),
            sortvg(12, 111, 11),
            sortvg(11, 12),
            sortvg(1, 11),

            string(0xd000, 'A'),
            string(0xd001, 'AB'),
            string(0xd002, 'AC'),
            string(0xd003, 'ACD'),
            string(0xd004, 'ACE'),
        ],
        newgrf=g,
    )


def test_lib_train_basics():
    a1 = DummyFeature(11)
    a2 = DummyFeature(12)
    b1 = DummyFeature(21)
    b2 = DummyFeature(22)
    b3 = DummyFeature(23)
    b4 = DummyFeature(24)
    c1 = DummyFeature(31)
    c2 = DummyFeature(32)
    c3 = DummyFeature(33)
    d1 = DummyFeature(41)
    d2 = DummyFeature(42)

    order = SetPurchaseOrder(
        VariantGroup(
            'A',
            a1,
            a2,
        ),
        VariantGroup(
            'B',
            b1,
            b2,
            b3,
            b4,
        ),
        VariantGroup(
            'C',
            c1,
            c2,
            c3,
        ),
        VariantGroup(
            'D',
            d1,
            d2,
        ),
    )

    check_lib(
        lambda g: order.set_variant_callbacks(g),
        order,
        [
            sortvg(42, None, 41),
            sortvg(41, 42),
            sortvg(33, 41, 31),
            sortvg(32, 33, 31),
            sortvg(31, 32),
            sortvg(24, 31, 21),
            sortvg(23, 24, 21),
            sortvg(22, 23, 21),
            sortvg(21, 22),
            sortvg(12, 21, 11),
            sortvg(11, 12),

            string(0xd000, 'A'),
            string(0xd001, 'B'),
            string(0xd002, 'C'),
            string(0xd003, 'D'),
        ],
    )

