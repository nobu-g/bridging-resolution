import pickle
import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import yaml
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Optional
import collections.abc

from kyoto_reader import KyotoReader
from pyknp import KNP, BList
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__file__)


def split_kc(input_dir: Path, output_dir: Path, max_subword_length: int, tokenizer: PreTrainedTokenizerBase):
    """
    各文書を，tokenize したあとの長さが max_subword_length 以下になるように複数の文書に分割する．
    1文に分割しても max_subword_length を超えるような長い文はそのまま出力する
    """
    did2sids: Dict[str, List[str]] = defaultdict(list)
    did2cumlens: Dict[str, List[int]] = {}
    sid2knp: Dict[str, str] = {}

    for knp_file in input_dir.glob('*.knp'):
        with knp_file.open() as fin:
            did = knp_file.stem
            did2cumlens[did] = [0]
            buff = ''
            for line in fin:
                buff += line
                if line.strip() == 'EOS':
                    blist = BList(buff)
                    did2sids[did].append(blist.sid)
                    did2cumlens[did].append(
                        did2cumlens[did][-1] + len(tokenizer.tokenize(' '.join(m.midasi for m in blist.mrph_list())))
                    )
                    sid2knp[blist.sid] = buff
                    buff = ''

    for did, sids in did2sids.items():
        cum: List[int] = did2cumlens[did]
        end = 1
        # end を探索
        while end < len(sids) and cum[end + 1] - cum[0] <= max_subword_length:
            end += 1

        idx = 0
        while end <= len(sids):
            start = 0
            # start を探索
            while start < end - 1 and cum[end] - cum[start] > max_subword_length:
                start += 1
            # start から end まで書き出し
            output_dir.joinpath(f'{did}-{idx:02}.knp').write_text(''.join(sid2knp[sid] for sid in sids[start:end]))
            idx += 1
            end += 1


def reparse_knp(knp_file: Path,
                output_dir: Path,
                knp: KNP,
                keep_dep: bool
                ) -> None:
    """係り受けなどを再付与"""
    blists: List[BList] = []
    with knp_file.open() as fin:
        buff = ''
        for line in fin:
            if line.startswith('+') or line.startswith('*'):
                if keep_dep is False:
                    buff += line[0] + '\n'  # ex) +
                else:
                    buff += ' '.join(line.split()[:2]) + '\n'  # ex) + 3D
            else:
                buff += line
            if line.strip() == 'EOS':
                blists.append(knp.reparse_knp_result(buff))
                buff = ''
    output_dir.joinpath(knp_file.name).write_text(''.join(blist.spec() for blist in blists))


def reparse(input_dir: Path,
            output_dir: Path,
            knp: KNP,
            bertknp: Optional[str] = None,
            n_jobs: int = 0,
            keep_dep: bool = False,
            ) -> None:
    if bertknp is None:
        args_iter = ((path, output_dir, knp, keep_dep) for path in input_dir.glob('*.knp'))
        if n_jobs > 0:
            with Pool(n_jobs) as pool:
                pool.starmap(reparse_knp, args_iter)
        else:
            for args in args_iter:
                reparse_knp(*args)
        return

    assert keep_dep is False, 'If you use BERTKNP, you cannot keep dependency labels.'

    buff = ''
    for knp_file in input_dir.glob('*.knp'):
        with knp_file.open() as fin:
            for line in fin:
                if line.startswith('+') or line.startswith('*'):
                    buff += line[0] + '\n'
                else:
                    buff += line

    out = subprocess.run(
        [
            bertknp,
            '-p',
            Path(bertknp).parents[1] / '.venv/bin/python',
            '-O',
            Path(__file__).parent.joinpath('bertknp_options.txt').resolve().__str__(),
            '-tab',
        ],
        input=buff, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
    )
    logger.warning(out.stderr)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_dir.joinpath('tmp.knp').write_text(out.stdout)
        reader = KyotoReader(tmp_dir / 'tmp.knp')
        for doc_id in reader.doc_ids:
            output_dir.joinpath(f'{doc_id}.knp').write_text(reader.get_knp(doc_id))


def process_kyoto(input_path: Path,
                  output_path: Path,
                  label: str,
                  do_reparse: bool,
                  n_jobs: int,
                  bertknp: Optional[str],
                  knp: KNP,
                  keep_dep: bool = False,
                  split: bool = False,
                  max_subword_length: int = None,
                  tokenizer: PreTrainedTokenizerBase = None,
                  ) -> int:
    """
    Process KNP format files in `input_path` and export them to `output_path`.
    This function does not explore files recursively.

    Args:
        input_path: An input directory.
        output_path: An output directory.
        label: Just a string to identify the kind of processing data.
        do_reparse: If True, perform feature addition and syntactic parsing again.
        n_jobs: The number of maximum processes. `-1` indicates the number of CPU cores.
        bertknp: If True, use BERTKNP instead of KNP.
        knp: A path to `knp`
        keep_dep: If True, keep already tagged dependencies when doing the reparse.
        split: If True, documents which contains more subword tokens than `max_subword_length` are split into multiple
         documents.
        max_subword_length: Used when `split` is True.
        tokenizer: A tokenizer to determine the number of subword tokens used when `split` is True.

    Returns:
        The number of resultant documents.

    """
    with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
        tmp_dir1, tmp_dir2 = Path(tmp_dir1), Path(tmp_dir2)
        if do_reparse is True:
            reparse(input_path, tmp_dir1, knp, bertknp=bertknp, n_jobs=n_jobs, keep_dep=keep_dep)
            input_path = tmp_dir1
        if split is True:
            # Because the length of the documents in KyotoCorpus is very long, split them into multiple documents
            # so that the tail sentence of each document has as much preceding sentences as possible.
            print('splitting corpus...')
            split_kc(input_path, tmp_dir2, max_subword_length, tokenizer)
            input_path = tmp_dir2

        output_path.mkdir(exist_ok=True)
        reader = KyotoReader(input_path, extract_nes=False, did_from_sid=(not split), n_jobs=n_jobs)
        for document in tqdm(reader.process_all_documents(), desc=label, total=len(reader)):
            with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
                pickle.dump(document, f)

    return len(reader)


def process_crowd(input_dir: Path,
                  out_dir: Path,
                  knp_dir: Path,
                  ) -> int:
    if input_dir.exists() is False:
        return 0
    out_knp_dir: Path = out_dir / 'crowd'
    out_knp_dir.mkdir(exist_ok=True)
    out_jsonl_dir: Path = out_dir / 'jsonl'
    out_jsonl_dir.mkdir(exist_ok=True)
    num_examples = 0
    for json_path in input_dir.iterdir():
        knp_path = knp_dir / f'{json_path.stem}.knp'
        if knp_path.exists() is True:
            shutil.copy(str(knp_path), str(out_knp_dir))
        else:
            print(f'knp file: {json_path.stem}.knp does not exist in {knp_dir} and ignored', file=sys.stderr)
            continue
        num_examples += 1
        shutil.copy(str(json_path), str(out_jsonl_dir))
    return num_examples


def merge_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kwdlc', type=str, default=None,
                        help='path to directory where KWDLC data exists')
    parser.add_argument('--kc', type=str, default=None,
                        help='path to directory where Kyoto Corpus data exists')
    parser.add_argument('--fuman', type=str, default=None,
                        help='path to directory where Fuman Corpus data exists')
    parser.add_argument('--jsonl-dir', type=str, required=True,
                        help='path to directory where jsonl data exists')
    parser.add_argument('--knp-dir', type=str, required=True,
                        help='path to directory where knp data exists')
    parser.add_argument('--out', type=(lambda p: Path(p)), required=True,
                        help='path to directory where dataset to be located')
    parser.add_argument('--max-seq-length', type=int, default=128,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--exophors', '--exo', default='著者,読者,不特定:人,不特定:物'.split(','), nargs='*',
                        help='exophor strings separated by ","')
    parser.add_argument('--bert-path', type=str, required=True,
                        help='path to pre-trained BERT model')
    parser.add_argument('--bert-name', type=str, required=True,
                        help='BERT model name')
    parser.add_argument('--jumanpp', type=str, default=shutil.which('jumanpp'),
                        help='path to jumanpp')
    parser.add_argument('--knp', type=str, default=shutil.which('knp'),
                        help='path to knp')
    parser.add_argument('--bertknp', type=str, default=None,
                        help='If specified, use BERTKNP instead of KNP for preprocessing.')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='number of processes of multiprocessing (default: number of cores)')
    args = parser.parse_args()

    # make directories to save dataset
    args.out.mkdir(exist_ok=True)
    exophors: List[str] = args.exophors
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, do_lower_case=False, tokenize_chinese_chars=False)

    knp = KNP(command=args.knp, jumancommand=args.jumanpp)
    # knp_case = KNP(command=args.knp, option='-tab -case2', jumancommand=args.jumanpp)

    config_path: Path = args.out / 'config.json'
    old_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config = {
        'path': str(args.out.resolve()),
        'max_seq_length': args.max_seq_length,
        'exophors': exophors,
        'bert_name': args.bert_name,
        'bert_path': args.bert_path,
        'vocab_size': tokenizer.vocab_size,
    }

    num_examples_dict = {}
    for ds in ('test', 'valid', 'train'):
        num_examples_ds = {}
        out_dir: Path = args.out / ds
        out_dir.mkdir(exist_ok=True)
        num_examples_ds['crowd'] = process_crowd(Path(args.jsonl_dir) / ds, out_dir, Path(args.knp_dir))
        kwargs = {
            'n_jobs': cpu_count() if args.n_jobs == -1 else args.n_jobs,
            'bertknp': args.bertknp,
            'knp': knp,
            'do_reparse': ds != 'train',
        }
        assert num_examples_ds['crowd'] == process_kyoto(out_dir / 'crowd', out_dir / 'crowd', f'{ds}/crowd', **kwargs)
        if ds in ('test', 'valid'):
            kwargs['do_reparse'] = False
            _ = process_kyoto(out_dir / 'crowd', out_dir / 'crowd_gold', f'{ds}/crowd_gold', **kwargs)

        for corpus in ('kwdlc', 'kc', 'fuman'):
            if getattr(args, corpus) is None:
                continue
            in_dir = Path(getattr(args, corpus)).resolve()
            out_corpus_dir: Path = out_dir / corpus
            out_corpus_dir.mkdir(exist_ok=True)
            kwargs = {
                'n_jobs': cpu_count() if args.n_jobs == -1 else args.n_jobs,
                'bertknp': args.bertknp,
                'knp': knp,
                'do_reparse': ds != 'train',
            }
            if corpus == 'kc':
                kwargs.update({
                    'split': True,
                    'max_subword_length': args.max_seq_length - len(exophors) - 3,  # [CLS], [SEP], [NULL]
                    'tokenizer': tokenizer,
                })

            num_examples_ds[corpus] = process_kyoto(in_dir / ds, out_corpus_dir, f'{ds}/{corpus}', **kwargs)

            if ds in ('test', 'valid'):
                kwargs['do_reparse'] = False
                if corpus == 'kc':
                    kwargs['split'] = False
                _ = process_kyoto(in_dir / ds, out_dir / f'{corpus}_gold', f'{ds}/{corpus}_gold', **kwargs)

        num_examples_dict[ds] = num_examples_ds

    config['num_examples'] = num_examples_dict

    merge_dict(old_config, config)

    with config_path.open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    yaml_config_path: Path = args.out / 'config.yaml'
    with yaml_config_path.open(mode='wt') as f:
        yaml.dump(config, f, indent=2, allow_unicode=True, sort_keys=False)


if __name__ == '__main__':
    main()
