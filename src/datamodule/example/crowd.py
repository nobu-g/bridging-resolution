import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

from dataclasses_json import dataclass_json
from kyoto_reader import Document

from .base import BaseExample

logger = logging.getLogger(__file__)


@dataclass_json
@dataclass(frozen=True)
class QuestionResult:
    """A dataclass to represent one question, which corresponds to one target phrase"""
    qid: str
    did: str
    target: Dict[str, Union[str, int]]  # keys: sid, tid, dtid
    votes: List[Dict[str, int]]
    null: int
    num_workers: int
    exophors: Dict[str, Dict[str, int]] = field(default_factory=dict)


class CrowdExample(BaseExample):
    """A single training/test example for bridging reference resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.annotation: Dict[int, QuestionResult] = {}  # bp level
        self._max_votes: List[int] = []

    @property
    def max_vote(self) -> float:
        return sum(self._max_votes) / len(self._max_votes)

    def load(self,
             document: Document,
             jsonl_path: Path,
             include_intra_sentential_cataphora: bool,
             ) -> None:
        dtid2votes: Dict[int, QuestionResult] = {}
        with jsonl_path.open() as f:
            for line in f:
                result: QuestionResult = QuestionResult.from_json(line.strip())
                dtid2votes[result.target['dtid']] = result
        self.doc_id = document.doc_id

        self._construct_phrases(document, include_intra_sentential_cataphora)

        for phrase in self.phrases:
            if phrase.is_target is True and phrase.dtid not in dtid2votes:
                # 唯一問題数の都合で全ての基本句がターゲットにならなかった文書のみ dtid2votes に存在しないことが起こりうる
                logger.warning(
                    f'gold votes for analysis target: {phrase.surf} (dtid: {phrase.dtid}, doc_id: {self.doc_id}) '
                    f'does not exist in {jsonl_path} and will be ignored')
                phrase.is_target = False
                for dmid in phrase.dmids:
                    self.mrphs[dmid].is_target = False

        for phrase in self.phrases:
            if phrase.is_target is False:
                continue
            ann = dtid2votes[phrase.dtid]
            self.annotation[phrase.dtid] = ann
            self._max_votes.append(ann.num_workers * 2)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ''
        for i, phrase in enumerate(self.phrases):
            string += f'{i:02} {phrase.surf}'
            if ann := self.annotation.get(i):
                pad = ' ' * (5 - len(phrase.surf)) * 2
                string += f'{pad}({" ".join("".join(map(str, vote.values())) for vote in ann.votes)})'
            string += '\n'
        return string
