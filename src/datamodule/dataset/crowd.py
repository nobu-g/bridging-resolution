from pathlib import Path
from typing import List, Optional, Union, Dict

from kyoto_reader import KyotoReader, Document
from omegaconf import ListConfig

from datamodule.example import CrowdExample
from .base import BaseDataset, LearningMethod, PhraseScore


class CrowdDataset(BaseDataset):
    def __init__(self,
                 knp_path: Union[str, Path],
                 jsonl_path: Union[str, Path],
                 max_seq_length: int,
                 bert_path: str,
                 exophors: ListConfig[str],
                 training: bool,
                 method: str,
                 include_intra_sentential_cataphora: bool,
                 gold_knp_path: Optional[str] = None,
                 n_jobs: int = -1,
                 logger=None,
                 ) -> None:
        super().__init__(knp_path, max_seq_length, bert_path, exophors, training, method, n_jobs, logger)
        self.jsonl_dir = Path(jsonl_path)
        self.include_intra_sentential_cataphora = include_intra_sentential_cataphora
        documents = list(self.reader.process_all_documents())
        self.examples: List[CrowdExample] = self._load(documents, str(knp_path))
        if not training:
            self.documents: Optional[List[Document]] = documents

        if self.method in (LearningMethod.REG,):
            self.max_score = sum(e.max_vote for e in self.examples) / len(self.examples)
        else:
            self.max_score = 1.0

        if not training:
            assert gold_knp_path is not None
            reader = KyotoReader(Path(gold_knp_path), target_cases=self.rels, extract_nes=False, n_jobs=n_jobs)
            self.gold_documents = list(reader.process_all_documents())
            self.gold_examples = self._load(self.gold_documents, gold_knp_path)

    def _load_example_from_document(self, document: Document) -> CrowdExample:
        example = CrowdExample()
        example.load(document, self.jsonl_dir / f'{document.doc_id}.jsonl', self.include_intra_sentential_cataphora)
        return example

    def _convert_example_to_scores(self,
                                   example: CrowdExample,
                                   ) -> List[PhraseScore]:
        phrase_scores = []
        for anaphor in example.phrases:
            if ann := example.annotation.get(anaphor.dtid):
                assert anaphor.is_target is True
                scores: List[float] = [-1 for _ in example.phrases]
                max_score = ann.num_workers * 2 if self.method in (LearningMethod.REG,) else 1.0
                for cand_dtid in anaphor.candidates:
                    vote: Dict[str, int] = ann.votes[cand_dtid]
                    score: float = max_score * self._vote2ratio(vote, num_workers=ann.num_workers)
                    scores[cand_dtid] = score

                exo_scores: List[float] = []
                for exophor in self.exophors:
                    assert exophor in ann.exophors, f'annotations for exophors do not exist in {self.jsonl_dir}'
                    vote = ann.exophors[exophor]
                    score: float = max_score * self._vote2ratio(vote, num_workers=ann.num_workers)
                    exo_scores.append(score)

                null_ratio: float = self._null2ratio(ann.null, num_workers=ann.num_workers)
                # 不特定:人, 不特定:物がデータについており，解析対象になっていなかった場合，NULL として扱う
                for exophor in ('不特定:人', '不特定:物'):
                    if exophor in ann.exophors and exophor not in self.exophors:
                        vote = ann.exophors[exophor]
                        null_ratio += self._vote2ratio(vote, num_workers=ann.num_workers)
                null_score: float = max_score * min(1.0, null_ratio)
                phrase_score = PhraseScore(scores, exo_scores, null_score)
            else:
                phrase_score = PhraseScore([], [], 0)
            phrase_scores.append(phrase_score)
        return phrase_scores

    @staticmethod
    def _vote2ratio(vote: Dict[str, int], num_workers: int) -> float:
        return (vote['1'] + vote['2'] * 2) / (num_workers * 2)

    @staticmethod
    def _null2ratio(null: int, num_workers: int) -> float:
        return null / num_workers
