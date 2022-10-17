import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

from kyoto_reader import Document
from tokenizers import Encoding

from utils.util import get_target_mask, get_candidate_mask

logger = logging.getLogger(__file__)


@dataclass
class Phrase:
    dtid: int
    surf: str
    dmids: List[int]
    dmid: int
    is_target: bool
    candidates: List[int]


@dataclass
class Mrph:
    dmid: int
    surf: str
    is_target: bool
    parent: Phrase


class LearningMethod(Enum):
    REG = 'regression'
    NREG = 'normalized-regression'
    SEL = 'selection'


class BaseExample:
    """A single training/test example for bridging reference resolution."""

    def __init__(self) -> None:
        self.example_id: int = -1
        self.phrases: List[Phrase] = []
        self.mrphs: List[Mrph] = []
        self.annotation: Dict[int, list] = {}  # bp level
        self.doc_id: str = ''
        self.encoding: Optional[Encoding] = None

    def _construct_phrases(self,
                           document: Document,
                           include_intra_sentential_cataphora: bool,
                           ):
        # construct phrases and mrphs
        bp_list = document.bp_list()
        bridging_target_mask: List[bool] = [mask for sent in document for mask in get_target_mask(sent.blist)]
        for anaphor in bp_list:
            bridging_candidate_mask: List[bool] = get_candidate_mask(
                anaphor,
                document,
                include_intra_sentential_cataphora=include_intra_sentential_cataphora,
            )
            candidates: List[int] = [bp.dtid for bp in bp_list if bridging_candidate_mask[bp.dtid]]
            is_target_phrase: bool = bridging_target_mask[anaphor.dtid]
            phrase = Phrase(
                dtid=anaphor.dtid,
                surf=anaphor.surf,
                dmids=anaphor.dmids,
                dmid=anaphor.dmid,
                is_target=is_target_phrase,
                candidates=candidates,
            )
            for morpheme in anaphor.mrph_list():
                dmid = document.mrph2dmid[morpheme]
                is_target_mrph = is_target_phrase and (dmid == phrase.dmid)
                mrph = Mrph(
                    dmid=dmid,
                    surf=morpheme.midasi,
                    is_target=is_target_mrph,
                    parent=phrase,
                )
                self.mrphs.append(mrph)
            self.phrases.append(phrase)

    def dump(self):
        data = {
            'phrases': [
                {
                    'surf': self.phrases[i].surf,
                    'dtid': i,
                    'scores': [
                        {
                            'dtid': j,
                            'phrase': self.phrases[j].surf,
                            'score': score
                        }
                        for j, score in enumerate(self.annotation[i][:-1]) if score > 0
                    ],
                    'null': self.annotation[i][-1] if self.phrases[i].is_target else None,
                }
                for i in range(len(self.phrases))
            ]
        }
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
