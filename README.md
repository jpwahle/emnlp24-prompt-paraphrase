# Paraphrase Types Elicit Prompt Engineering Capabilities
[![arXiv](https://img.shields.io/badge/arXiv-2310.14863-b31b1b.svg)](https://arxiv.org/abs/2310.14863)

![Under Construction](./teaser_method.png)

## Overview
This repository provides the implementation of the paper "Paraphrase Types Elicit Prompt Engineering Capabilities".

## Data
The main tasks of this study are from the [SuperNatural-Instruction](https://github.com/allenai/natural-instructions) dataset, which is available under `/data`.

## Scripts and Usage

### Paraphrase Prompts and Run Tasks
Run the main script `paraphrase_and_run_tasks.py` to generate paraphrases and execute tasks.

### Index FineWeb Dataset
Use the `build_bm25_index.py` script to index the FineWeb corpus using the BM25 algorithm.

### Analyzing the Results
To compute gains and losses, and measure lexical diversity, use `analysis_gain_loss_lexical_diversity.py`.

Evaluate the complexity of prompts with `analysis_complexity.py`.

Assess the closeness of prompts to the training data using `analysis_closeness_to_training.py`.

## Contributing
You can contribute by reviewing source code changes and adding new features.

## Citation

```bib
@inproceedings{wahle-etal-2024-paraphrase,
    title = "Paraphrase Types Elicit Prompt Engineering Capabilities",
    author = "Wahle, Jan Philip  and
      Ruas, Terry  and
      Xu, Yang  and
      Gipp, Bela",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.617",
    pages = "11004--11033",
    abstract = "Much of the success of modern language models depends on finding a suitable prompt to instruct the model. Until now, it has been largely unknown how variations in the linguistic expression of prompts affect these models. This study systematically and empirically evaluates which linguistic features influence models through paraphrase types, i.e., different linguistic changes at particular positions. We measure behavioral changes for five models across 120 tasks and six families of paraphrases (i.e., morphology, syntax, lexicon, lexico-syntax, discourse, and others). We also control for other prompt engineering factors (e.g., prompt length, lexical diversity, and proximity to training data). Our results show a potential for language models to improve tasks when their prompts are adapted in specific paraphrase types (e.g., 6.7{\%} median gain in Mixtral 8x7B; 5.5{\%} in LLaMA 3 8B). In particular, changes in morphology and lexicon, i.e., the vocabulary used, showed promise in improving prompts. These findings contribute to developing more robust language models capable of handling variability in linguistic expression.",
}
```

## License
Licensed under the Apache 2.0 license.
