import json
from pathlib import Path
from typing import List

from datamodule.dataset import CrowdDataset, KyotoDataset
from datamodule.dataset.base import PhraseScore


def test_crowd_dataset(fixture_crowd_dataset: CrowdDataset, fixture_expected_dir: Path):
    example = fixture_crowd_dataset.examples[0]
    expected = json.loads(fixture_expected_dir.joinpath('crowd_dataset0.json').read_text())
    assert len(example.phrases) == len(expected['phrases'])

    phrase_scores: List[PhraseScore] = fixture_crowd_dataset.to_phrase_scores(example)
    for i in range(len(example.phrases)):
        phrase = example.phrases[i]
        expected_phrase = expected['phrases'][i]
        assert expected_phrase['surf'] == phrase.surf
        assert expected_phrase['dtid'] == i
        if phrase.is_target is True:
            phrase_score = phrase_scores[phrase.dtid]
            assert len(phrase_score.phrases) == len(example.phrases)
            assert len(phrase_score.exophors) == 0
            for score in expected_phrase['scores']:
                assert phrase_score.phrases[score['dtid']] == score['score']
            assert phrase_score.null == expected_phrase['null']
        else:
            assert expected_phrase['null'] is None


def test_kyoto_dataset(fixture_kyoto_dataset: KyotoDataset, fixture_expected_dir: Path):
    example = fixture_kyoto_dataset.examples[0]
    expected = json.loads(fixture_expected_dir.joinpath('kyoto_dataset0.json').read_text())
    assert len(example.phrases) == len(expected['phrases'])

    phrase_scores: List[PhraseScore] = fixture_kyoto_dataset.to_phrase_scores(example)
    for i in range(len(example.phrases)):
        phrase = example.phrases[i]
        expected_phrase = expected['phrases'][i]
        assert expected_phrase['surf'] == phrase.surf
        assert expected_phrase['dtid'] == i
        if phrase.is_target is True:
            phrase_score = phrase_scores[phrase.dtid]
            assert len(phrase_score.phrases) == len(example.phrases)
            assert len(phrase_score.exophors) == 2
            for score in expected_phrase['scores']:
                assert phrase_score.phrases[score['dtid']] == score['score']
            assert phrase_score.null == expected_phrase['null']
        else:
            assert expected_phrase['null'] is None
