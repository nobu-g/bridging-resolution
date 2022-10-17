import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, asdict
from logging import Logger
from pathlib import Path
from typing import List, Dict, Union, Tuple

from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
from kyoto_reader import KyotoReader, Document
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from tokenizers import Encoding
from omegaconf import ListConfig

from datamodule.example.base import BaseExample, LearningMethod


@dataclass(frozen=True)
class InputFeatures:
    """
    A dataclass which represents a raw model input.
    The attributes of this class correspond to arguments of forward method of each model.
    """
    example_id: int
    input_ids: List[int]
    attention_mask: List[bool]
    segment_ids: List[int]
    mask: List[List[bool]]
    target: List[List[float]]


@dataclass(frozen=True)
class PhraseScore:
    phrases: List[float]
    exophors: List[float]
    null: float


class BaseDataset(Dataset):
    def __init__(self,
                 knp_path: Union[str, Path],
                 max_seq_length: int,
                 bert_path: str,
                 exophors: ListConfig[str],
                 training: bool,
                 method: str,  # regression, norm-regression, or selection
                 n_jobs: int,
                 logger=None,
                 ) -> None:
        self.knp_path = Path(knp_path)
        self.max_seq_length: int = max_seq_length
        self.exophors: List[str] = list(exophors)
        self.special_tokens: List[str] = self.exophors + ['NULL']
        self.special_to_index: Dict[str, int] = {
            token: max_seq_length - i - 1 for i, token in enumerate(reversed(self.special_tokens))
        }
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            bert_path,
            do_lower_case=False,
            additional_special_tokens=self.special_tokens,
            tokenize_chinese_chars=False
        )
        self.bert_path: str = bert_path
        self.training: bool = training
        self.method: LearningMethod = LearningMethod(method)
        self.logger: Logger = logger or logging.getLogger(__file__)
        self.examples: List[BaseExample] = []
        self.max_score = None
        self.rels: List[str] = ['ノ', 'ノ？', '修飾']
        self.reader = KyotoReader(self.knp_path, target_cases=self.rels, extract_nes=False, n_jobs=0)

    @property
    def special_indices(self) -> List[int]:
        return list(self.special_to_index.values())

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    def _load(self, documents: List[Document], documents_path: str):
        examples = []
        load_cache: bool = ('BRG_DISABLE_CACHE' not in os.environ and 'BRG_OVERWRITE_CACHE' not in os.environ)
        save_cache: bool = ('BRG_DISABLE_CACHE' not in os.environ)
        bpa_cache_dir: Path = Path(os.environ.get('BRG_CACHE_DIR', f'/data/{os.environ["USER"]}/brg_cache'))
        for document in tqdm(documents, desc='processing documents'):
            # give enough options to identify examples
            hash_ = self._hash(document, documents_path, str(self.bert_path))
            example_cache_path = bpa_cache_dir / hash_ / f'{document.doc_id}.pkl'
            if example_cache_path.exists() and load_cache:
                with example_cache_path.open('rb') as f:
                    try:
                        example = pickle.load(f)
                    except EOFError:
                        example = self._load_example_from_document(document)
            else:
                example = self._load_example_from_document(document)
                if save_cache:
                    self._save_cache(example, example_cache_path)
            examples.append(example)
        examples = self._post_process_examples(examples)
        if len(examples) == 0:
            self.logger.error('No examples to process. '
                              f'Make sure there exist any documents in {self.knp_path} and they are not too long.')
        return examples

    @pl.utilities.rank_zero_only
    def _save_cache(self, example, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open('wb') as f:
            pickle.dump(example, f)

    @staticmethod
    def _hash(document, *args) -> str:
        attrs = ('relax_cases', 'extract_nes', 'use_pas_tag')
        assert set(attrs) <= set(vars(document).keys())
        vars_document = {k: v for k, v in vars(document).items() if k in attrs}
        string = repr(sorted(vars_document)) + ''.join(repr(a) for a in args)
        return hashlib.md5(string.encode()).hexdigest()

    def _post_process_examples(self, examples: List[BaseExample]) -> List[BaseExample]:
        idx = 0
        filtered = []
        for example in examples:
            encoding: Encoding = self.tokenizer(
                [m.surf for m in example.mrphs],
                is_split_into_words=True,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - self.num_special_tokens:
                continue
            example.encoding = encoding
            example.example_id = idx
            filtered.append(example)
            idx += 1
        return filtered

    def _load_example_from_document(self, document: Document) -> BaseExample:
        raise NotImplementedError

    def to_phrase_scores(self, example: BaseExample) -> List[PhraseScore]:
        return self._convert_example_to_scores(example)

    def dump_prediction(self,
                        result: List[List[float]],  # subword level
                        example: BaseExample,
                        ) -> List[List[float]]:  # phrase level
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．shape は (n_phrase, 0 or n_phrase+special)．"""
        ret = [[] for _ in example.phrases]
        for anaphor in example.phrases:
            if anaphor.is_target is False:
                continue
            # src 側は先頭のトークンを採用
            token_index_span: Tuple[int, int] = example.encoding.word_to_tokens(anaphor.dmid)
            token_level_votes: List[float] = result[token_index_span[0]]
            bp_level_votes: List[float] = []
            for tgt_bp in example.phrases:
                if tgt_bp.dtid not in anaphor.candidates:
                    bp_level_votes.append(-1)  # pad -1 for non-candidate phrases
                    continue
                # tgt 側は複数のサブワードから構成されるため平均を取る
                token_index_span: Tuple[int, int] = example.encoding.word_to_tokens(tgt_bp.dmid)
                votes = token_level_votes[slice(*token_index_span)]
                bp_level_votes.append(sum(votes) / len(votes))
            bp_level_votes += [token_level_votes[idx] for idx in self.special_indices]
            assert len(bp_level_votes) == len(example.phrases) + len(self.special_to_index)
            ret[anaphor.dtid] = bp_level_votes
        return ret

    def dump_gold(self,
                  example: BaseExample,
                  ) -> List[List[float]]:
        """1 example 中に存在する基本句それぞれに対して正解のリストを返す．shape は (n_phrase, 0 or n_phrase+special)．"""
        ret = [[] for _ in example.phrases]
        for anaphor, phrase_score in zip(example.phrases, self._convert_example_to_scores(example)):
            if anaphor.is_target is False:
                continue
            bp_level_votes = phrase_score.phrases + phrase_score.exophors + [phrase_score.null]  # -1 for non-candidate
            ret[anaphor.dtid] = bp_level_votes
        return ret

    def _convert_example_to_scores(self, example: BaseExample) -> List[PhraseScore]:
        raise NotImplementedError

    def _convert_example_to_feature(self,
                                    example: BaseExample,
                                    ) -> InputFeatures:
        """Loads a data file into a list of `InputBatch`s."""

        max_seq_length = self.max_seq_length

        scores_set: List[List[float]] = [[0] * max_seq_length for _ in range(max_seq_length)]
        candidates_set: List[List[int]] = [[] for _ in range(max_seq_length)]

        phrase_scores: List[PhraseScore] = self._convert_example_to_scores(example)

        for word in example.mrphs:
            if word.is_target is False:
                continue
            assert word.parent is not None
            phrase = word.parent
            phrase_score = phrase_scores[phrase.dtid]
            special_scores: List[float] = phrase_score.exophors + [phrase_score.null]

            scores: List[float] = [0] * max_seq_length
            candidates: List[int] = []
            for dtid in phrase.candidates:
                score: float = phrase_score.phrases[dtid]
                assert score >= 0
                ph = example.phrases[dtid]
                # 対象形態素を構成する全サブワードに対して票数を予測させる
                token_index_span: Tuple[int, int] = example.encoding.word_to_tokens(ph.dmid)
                for tkid in range(*token_index_span):
                    scores[tkid] = score
                    candidates.append(tkid)
            for special_score, tkid in zip(special_scores, self.special_to_index.values()):
                scores[tkid] = special_score
                candidates.append(tkid)

            token_index_span: Tuple[int, int] = example.encoding.word_to_tokens(word.dmid)
            # target 側は先頭のサブワードを採用
            scores_set[token_index_span[0]] = scores
            candidates_set[token_index_span[0]] = candidates

        special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

        merged_encoding: Encoding = Encoding.merge([example.encoding, special_encoding])

        # ensure all features are padded to max_seq_length
        for feature in (scores_set, candidates_set):
            assert len(feature) == max_seq_length

        feature = InputFeatures(
            example_id=example.example_id,
            input_ids=merged_encoding.ids,
            attention_mask=merged_encoding.attention_mask,
            segment_ids=merged_encoding.type_ids,
            mask=[
                [(x in cands) for x in range(max_seq_length)] for cands in candidates_set
            ],  # False -> mask, True -> keep
            target=scores_set,
        )

        return feature

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        feature = self._convert_example_to_feature(self.examples[idx])
        return asdict(feature)
