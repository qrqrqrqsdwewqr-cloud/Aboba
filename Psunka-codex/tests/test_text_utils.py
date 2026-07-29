import config
from text_utils import normalize_transcript_text, trim_repeated_prefix


def test_che_replacement_without_dots():
    assert normalize_transcript_text("че происходит", config.REPLACEMENTS, config.YO_WORD_REPLACEMENTS) == "что происходит"


def test_yo_replacements_are_applied():
    assert normalize_transcript_text("еще все ее самолет", config.REPLACEMENTS, config.YO_WORD_REPLACEMENTS) == "ещё всё её самолёт"


def test_trim_repeated_prefix_from_previous_fragment_tail():
    assert trim_repeated_prefix("наш лето началось", "мы проводим наш лето", 4) == "началось"
