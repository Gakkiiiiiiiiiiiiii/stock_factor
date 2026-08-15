from stock_factor.application.seed_library import Alpha191SeedLibrary
from stock_factor.ports.symbols import normalize_symbol, symbol_exchange, symbol_local_code


def test_normalize_cross_service_symbols():
    assert normalize_symbol("600519.SH") == "CN.A.600519"
    assert normalize_symbol("00700.HK") == "HK.00700"
    assert normalize_symbol("NVDA") == "US.NVDA"
    assert symbol_exchange("HK.00700") == "HK"
    assert symbol_local_code("HK.00700") == "00700"


def test_alpha191_catalog_restores_full_baseline_size():
    assert len(Alpha191SeedLibrary().load()) == 44
