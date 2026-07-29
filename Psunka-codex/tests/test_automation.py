from automation import category_key, sanitize_transcript_text


def test_sanitize_removes_category_digit_prefix():
    assert sanitize_transcript_text("1это только там") == "это только там"


def test_sanitize_removes_gui_prefixes():
    assert sanitize_transcript_text("[15:39:51] [Category: 1] это только там") == "это только там"


def test_category_key_uses_configured_table():
    assert category_key("speech") == "2"
    assert category_key(4) == "4"


def test_unknown_category_has_no_key():
    assert category_key("unknown_category") is None
