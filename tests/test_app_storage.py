import importlib


def test_storage_sanitizes_filename_and_clears_history(tmp_path, monkeypatch):
    monkeypatch.setenv("CASTING_INSPECTION_APP_DATA", str(tmp_path / "user-data"))
    import src.inference.app_storage as storage
    storage = importlib.reload(storage)
    storage.add_inspection(r"C:\\private\\customer-part.jpg", "REVIEW", "crack", 0.42)
    rows = storage.get_recent_inspections()
    assert len(rows) == 1
    assert rows[0]["image_name"] == "customer-part.jpg"
    storage.clear_inspections()
    assert storage.get_recent_inspections() == []
