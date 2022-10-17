# bridging-reference: A Japanese bridging reference resolver
[[Paper]](https://aclanthology.org/2022.crac-1.8/)

## Description

This project provides a system to perform Japanese bridging reference resolution.

## Requirements

- Python 3.8+
- [Juman++](https://github.com/ku-nlp/jumanpp) 2.0.0-rc3
- [KNP](https://github.com/ku-nlp/knp) 5.0

## Setup Environment

### Python Virtual Environment

`poetry install`

## Training A Model From Scratch

### Downloading Corpora

```shell
cd /somewhere
```
- KWDLC

  ```shell
  mkdir kwdlc
  git clone https://github.com/ku-nlp/KWDLC kwdlc/KWDLC
  ```

### Adding features

[kyoto-reader](https://github.com/ku-nlp/kyoto-reader), which this project depends on,
provides [some commands](https://kyoto-reader.readthedocs.io/en/latest/#corpus-preprocessor) to preprocess corpora.
For example, `kyoto configure` creates `Makefile` to add features for this analyzer,
and `kyoto idsplit` splits corpus documents into train/valid/test datasets.

```shell
$ git clone https://github.com/ku-nlp/JumanDIC
$ kyoto configure --corpus-dir /somewhere/kwdlc/KWDLC/knp \
--data-dir /somewhere/kwdlc \
--juman-dic-dir /somewhere/JumanDIC/dic
created Makefile at /somewhere/kwdlc
$ cd /somewhere/kwdlc && make -i
$ kyoto idsplit --corpus-dir /somewhere/kwdlc/knp \
--output-dir /somewhere/kwdlc \
--train /somewhere/kwdlc/KWDLC/id/split_for_pas/train.id \
--valid /somewhere/kwdlc/KWDLC/id/split_for_pas/dev.id \
--test /somewhere/kwdlc/KWDLC/id/split_for_pas/test.id
```

### Preprocessing Documents

After adding features to the corpora, you need to load and pickle them.

```shell
python src/preprocess.py \
--kwdlc /somewhere/kwdlc \
--kc /somewhere/kc \
--out /somewhere/dataset \
--bert-name roberta \
--bert-path nlp-waseda/roberta-base-japanese
```

`src/preprocess.py` creates a file `config.yml` in which preprocessing settings and statistics of the dataset are written.
Copy this file to the config directory.

```shell
cp data/bldsample/config.yaml conf/dataset/example.yaml
```

### Running training

- Copy `conf/example.yaml` and edit the contents.

    ```shell
    cp conf/example.yaml conf/custom.yaml
    ```
  
- Run training
  
    ```shell
    python src/train.py -cn custom devices=[0,1] num_workers=4
    ```

## Running testing

```shell
poetry run python src/test.py checkpoint=/path/to/trained/checkpoint eval_set=valid devices=[0,1]
```

## Environment Variables

- `BRG_CACHE_DIR`: A directory where processed documents are cached. Default value is `/data/$USER/brg_cache`.
- `BRG_OVERWRITE_CACHE`: If set, the data loader does not load cache even if it exists.
- `BRG_DISABLE_CACHE`: If set, the data loader does not load or save cache.
- `WANDB_PROJECT`: Project name in Weights & Biases.

## Dataset

- Kyoto University Web Document Leads Corpus ([KWDLC](https://github.com/ku-nlp/KWDLC))

## Citation

```bibtex
@inproceedings{ueda-kurohashi-2022-improving,
    title = "Improving Bridging Reference Resolution using Continuous Essentiality from Crowdsourcing",
    author = "Ueda, Nobuhiro  and
      Kurohashi, Sadao",
    booktitle = "Proceedings of the Fifth Workshop on Computational Models of Reference, Anaphora and Coreference",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.crac-1.8",
    pages = "74--87",
    abstract = "Bridging reference resolution is the task of finding nouns that complement essential information of another noun. The essentiality varies depending on noun combination and context and has a continuous distribution. Despite the continuous nature of essentiality, existing datasets of bridging reference have only a few coarse labels to represent the essentiality. In this work, we propose a crowdsourcing-based annotation method that considers continuous essentiality. In the crowdsourcing task, we asked workers to select both all nouns with a bridging reference relation and a noun with the highest essentiality among them. Combining these annotations, we can obtain continuous essentiality. Experimental results demonstrated that the constructed dataset improves bridging reference resolution performance. The code is available at https://github.com/nobu-g/bridging-resolution.",
}
```
## Licence

MIT

## Author

Nobuhiro Ueda <ueda **at** nlp.ist.i.kyoto-u.ac.jp>
