from pathlib import Path
from typing import List, Optional, Union

from kyoto_reader import KyotoReader, Document
from omegaconf import ListConfig

from datamodule.example import KyotoExample
from .base import BaseDataset, LearningMethod, PhraseScore


class KyotoDataset(BaseDataset):
    def __init__(self,
                 knp_path: Union[str, Path],
                 max_seq_length: int,
                 bert_path: str,
                 exophors: ListConfig[str],
                 training: bool,
                 method: str,
                 kc: bool,
                 include_intra_sentential_cataphora: bool = False,
                 gold_knp_path: Optional[str] = None,
                 n_jobs: int = -1,
                 logger=None,
                 ) -> None:
        super().__init__(knp_path, max_seq_length, bert_path, exophors, training, method, n_jobs, logger)
        self.kc: bool = kc
        self.include_intra_sentential_cataphora: bool = include_intra_sentential_cataphora
        self.max_score = 16.0 if self.method in (LearningMethod.REG,) else 1.0

        documents = list(self.reader.process_all_documents())
        self.examples: List[KyotoExample] = self._load(documents, str(knp_path))
        if not training:
            self.documents: Optional[List[Document]] = documents

        if not training:
            assert gold_knp_path is not None
            reader = KyotoReader(Path(gold_knp_path), target_cases=self.rels, extract_nes=False, n_jobs=n_jobs)
            self.gold_documents = list(reader.process_all_documents())
            self.gold_examples = self._load(self.gold_documents, gold_knp_path)

    def _load_example_from_document(self, document: Document) -> KyotoExample:
        example = KyotoExample()
        example.load(document, self.include_intra_sentential_cataphora)
        return example

    def _convert_example_to_scores(self,
                                   example: KyotoExample,
                                   ) -> List[PhraseScore]:
        phrase_scores = []
        for anaphor in example.phrases:
            if ann := example.annotation.get(anaphor.dtid):
                scores: List[float] = [-1 for _ in example.phrases]
                for cand_dtid in anaphor.candidates:
                    rel: Optional[str] = ann.phrase_relations.get(cand_dtid)
                    score: float = self.max_score * self._rel2ratio(rel)
                    scores[cand_dtid] = score

                exo_scores: List[float] = []
                for exophor in self.exophors:
                    rel: Optional[str] = ann.exophor_relations.get(exophor)
                    score: float = self.max_score * self._rel2ratio(rel)
                    exo_scores.append(score)

                rels: List[str] = list(ann.phrase_relations.values()) + list(ann.exophor_relations.values())
                null_ratio: float = self._rel2ratio_null(rels)
                # 不特定:人, 不特定:物がデータについており，解析対象になっていなかった場合，NULL として扱う
                for exophor in ('不特定:人', '不特定:物'):
                    if exophor in ann.exophor_relations \
                        and exophor not in self.exophors \
                        and sum(scores + exo_scores) == 0:
                        rel = ann.exophor_relations[exophor]
                        null_ratio += self._rel2ratio(rel)
                null_score: float = self.max_score * min(1.0, null_ratio)
                phrase_score = PhraseScore(scores, exo_scores, null_score)
            else:
                phrase_score = PhraseScore([], [], 0)
            phrase_scores.append(phrase_score)
        return phrase_scores

    @staticmethod
    def _rel2ratio(rel: Optional[str]) -> float:
        ratio: float = 0.0
        if rel == 'ノ':
            ratio = 1.0
        elif rel == 'ノ？':
            ratio = 0.5
        elif rel == '修飾':
            ratio = 0.25
        return ratio

    @staticmethod
    def _rel2ratio_null(rels: List[str]) -> float:
        ratio: float = 1.0
        if 'ノ' in rels:
            ratio = 0.0
        elif 'ノ？' in rels:
            ratio = 0.5
        elif '修飾' in rels:
            ratio = 0.75
        return ratio
