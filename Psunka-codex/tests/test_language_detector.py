from language_detector import detect_language

def test_detect_ru_en_unknown():
    assert detect_language('привет мир').language == 'ru'
    assert detect_language('hello world').language == 'en'
    assert detect_language('да').language == 'unknown'
