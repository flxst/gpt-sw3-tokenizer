# gpt-sw3-tokenizer

Train, evaluate and analyze BPE tokenizers.

A paper associated with this repository can be found [here](https://arxiv.org/abs/2304.14780).

## Resources

* source code: [https://github.com/flxst/gpt-sw3-tokenizer](https://github.com/flxst/gpt-sw3-tokenizer)
* documentation: [https://flxst.github.io/gpt-sw3-tokenizer](https://flxst.github.io/gpt-sw3-tokenizer)

## Installation

``` bash
git clone https://github.com/flxst/gpt-sw3-tokenizer.git
pip install -r requirements.txt
```

-----------
## About

This repository provides easy-to-use tools to sample (weighted) data and subsequently train, evaluate and analyze a tokenizer.

<div align="center">
<img alt="sampling" src="docs/docs/images/filter-solid-margin.png" height="53">&nbsp;
&nbsp;
&nbsp;
&nbsp;
<img alt="training" src="docs/docs/images/brain-solid-margin.png" height="53">&nbsp;
&nbsp;
&nbsp;
&nbsp;
<img alt="evaluation" src="docs/docs/images/ruler-solid-margin.png" height="53">&nbsp;
&nbsp;
&nbsp;
&nbsp;
<img alt="analysis" src="docs/docs/images/magnifying-glass-solid-margin.png" height="53">&nbsp;
</div>
<div align="center">
<a href="sampling">Sampling</a>
&nbsp;
&nbsp;
&nbsp;
<a href="training">Training</a>
&nbsp;
&nbsp;
&nbsp;
<a href="evaluation">Evaluation</a>
&nbsp;
&nbsp;
&nbsp;
<a href="analysis">Analysis</a>
&nbsp;
</div>

## Features

<img src="docs/docs/images/filter-solid-margin.png" height="13">&nbsp;[Sampling](sampling.md)

- weighting of different categories and languages

<img src="docs/docs/images/brain-solid-margin.png" height="13">&nbsp;[Training](training.md)

- support for SentencePiece and HuggingFace
- customizable tokenizer features (vocabulary size, handling of whitespace and numbers, ..)

<img src="docs/docs/images/ruler-solid-margin.png" height="13">&nbsp;[Evaluation](evaluation.md)

- computation of common tokenizer metrics (unknown rate, fertility, proportion of continued words, ..)

<img src="docs/docs/images/magnifying-glass-solid-margin.png" height="13">&nbsp;[Analysis](analysis.md)

- example tokenization
- vocabulary overlap and performance comparison across languages
- effect of the vocabulary size

## Citation

``` tex
@misc{gpt-sw3-tokenizer,
  title = {Training and Evaluation of a Multilingual Tokenizer for {GPT}-{SW3}},
  url = {http://arxiv.org/abs/2304.14780},
  author = {Stollenwerk, Felix},
  year = {2023},
}
```

-----------
