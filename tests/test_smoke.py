"""These pass on day one. They check the plumbing so you notice quickly if you break it."""
from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_the_page_file_is_valid_python():
    import py_compile
    py_compile.compile("app/main.py", doraise=True)
