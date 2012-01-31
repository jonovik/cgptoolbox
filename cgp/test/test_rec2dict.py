"""Tests for :mod:`cgp.utils.rec2dict`."""

from ..utils import rec2dict

def test_dict2rec_unicode():
    """Test with unicode keys."""
    rec2dict.dict2rec({u"a": u"b"})
