from stock_factor.ports.symbols import normalize_symbol


def test_normalize_cross_service_symbols():
    assert normalize_symbol("600519.SH") == "CN.A.600519"
    assert normalize_symbol("00700.HK") == "HK.00700"
    assert normalize_symbol("NVDA") == "US.NVDA"
