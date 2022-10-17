import os
import sys
from pathlib import Path

from omegaconf import ListConfig
import pytest

here = Path(__file__).parent
sys.path.append(str(here.parent / 'src'))

from datamodule.dataset import CrowdDataset, KyotoDataset
from datamodule.example.base import LearningMethod

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@pytest.fixture()
def fixture_crowd_dataset() -> CrowdDataset:
    dataset = CrowdDataset(
        here / 'data/knp',
        here / 'data/jsonl',
        max_seq_length=128,
        bert_path=os.environ['BERT_PATH'],
        exophors=ListConfig([]),
        training=True,
        method=LearningMethod.REG.value,
        include_intra_sentential_cataphora=True,
    )
    return dataset


@pytest.fixture()
def fixture_kyoto_dataset() -> KyotoDataset:
    dataset = KyotoDataset(
        here / 'data' / 'kwdlc',
        max_seq_length=128,
        bert_path=os.environ['BERT_PATH'],
        exophors=ListConfig(['著者', '読者']),
        training=True,
        method=LearningMethod.REG.value,
        kc=False,
        include_intra_sentential_cataphora=True,
    )
    return dataset


@pytest.fixture()
def fixture_expected_dir():
    return here.joinpath('data/expected').resolve()
