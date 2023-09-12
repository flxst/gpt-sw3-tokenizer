# gpt-sw3-tokenizer

Train, evaluate and analyze BPE tokenizers.

A paper associated with this repository can be found [here](https://arxiv.org/abs/2304.14780).

-----------
## Resources

* source code: [https://github.com/flxst/gpt-sw3-tokenizer](https://github.com/flxst/gpt-sw3-tokenizer)
* documentation: [https://flxst.github.io/gpt-sw3-tokenizer](https://flxst.github.io/gpt-sw3-tokenizer)

-----------
## Installation

``` bash
git clone https://github.com/flxst/gpt-sw3-tokenizer.git
pip install -r requirements.txt
```

-----------
## About

This repository provides easy-to-use tools to sample (weighted) data and subsequently train, evaluate and analyze a tokenizer.

<div align="left">
<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/filter.svg" height="50">&nbsp;
&nbsp;
&nbsp;
&nbsp;
<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/brain.svg" height="50">&nbsp;
&nbsp;
&nbsp;
&nbsp;
<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/ruler.svg" height="50">&nbsp;
&nbsp;
&nbsp;
&nbsp;
<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/magnifying-glass.svg" height="50">&nbsp;
</div>
<div align="left">
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
</div>

-----------
## Features

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/filter.svg" height="13">&nbsp;[Sampling](sampling.md)

- weighting of different categories and languages

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/brain.svg" height="13">&nbsp;[Training](training.md)

- support for SentencePiece and HuggingFace
- customizable tokenizer features (vocabulary size, handling of whitespace and numbers, ..)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/ruler.svg" height="13">&nbsp;[Evaluation](evaluation.md)

- computation of common tokenizer metrics (unknown rate, fertility, proportion of continued words, ..)

<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/magnifying-glass.svg" height="13">&nbsp;[Analysis](analysis.md)

- example tokenization
- vocabulary overlap and performance comparison across languages
- effect of the vocabulary size

-----------
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
