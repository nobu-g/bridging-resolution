import logging
from dataclasses import dataclass
from typing import List, Dict

from kyoto_reader import Document, Argument, SpecialArgument

from .base import BaseExample

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class KyotoAnnotation:
    phrase_relations: Dict[int, str]
    exophor_relations: Dict[str, str]


class KyotoExample(BaseExample):
    """A single training/test example for bridging reference resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.annotation: Dict[int, KyotoAnnotation] = {}  # bp level
        exophors: List[str] = ['著者', '読者', '不特定:人', '不特定:物']
        self.relax_exophors = {}
        for exophor in exophors:
            self.relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in '１２３４５６７８９':
                    self.relax_exophors[exophor + n] = exophor

    def load(self,
             document: Document,
             include_intra_sentential_cataphora: bool,
             ) -> None:
        bp_list = document.bp_list()
        self.doc_id = document.doc_id

        self._construct_phrases(document, include_intra_sentential_cataphora)

        for anaphor, phrase in zip(bp_list, self.phrases):
            if phrase.is_target is False:
                continue
            phrase_relations: Dict[int, str] = {}
            exophor_relations: Dict[str, str] = {}
            for rel, args in document.get_arguments(anaphor, relax=True).items():
                for arg in args:
                    if isinstance(arg, Argument) is True:
                        if arg.dtid not in phrase.candidates:
                            continue
                        phrase_relations[arg.dtid] = rel
                    else:
                        assert isinstance(arg, SpecialArgument)
                        if arg.exophor not in self.relax_exophors:
                            continue
                        exophor_relations[self.relax_exophors[arg.exophor]] = rel

            self.annotation[phrase.dtid] = KyotoAnnotation(phrase_relations, exophor_relations)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ''
        for i, phrase in enumerate(self.phrases):
            string += f'{i:02} {phrase.surf}'
            if ann := self.annotation.get(i):
                pad = ' ' * (5 - len(phrase.surf)) * 2
                string += f'{pad}({" ".join(f"{vote}" for vote in ann.phrase_relations)})'
            string += '\n'
        return string
