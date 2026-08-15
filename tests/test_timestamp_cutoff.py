from stock_factor.application.panel import _available_from


def test_available_from_keeps_timezone_for_cutoff_policy():
    timestamp = _available_from({"available_from": "2026-08-15T09:20:00+08:00"})
    assert timestamp is not None
    assert timestamp.hour == 9 and timestamp.minute == 20
