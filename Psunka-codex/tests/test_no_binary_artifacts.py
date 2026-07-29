from pathlib import Path


def test_repository_does_not_commit_binary_templates():
    pngs = sorted(Path('templates').glob('*.png'))
    assert pngs == []
