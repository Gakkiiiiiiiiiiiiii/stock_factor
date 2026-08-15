from stock_factor.engine.oos_audit import audit_final_oos


def test_oos_audit_rejects_missing_split_and_snapshot():
    audit = audit_final_oos(split=None, final_oos={"passed": True}, data_snapshot_id=None)
    assert audit["audit_status"] == "FAILED"
    assert {"RESEARCH_SPLIT_MISSING", "DATA_SNAPSHOT_MISSING"}.issubset(audit["violations"])
