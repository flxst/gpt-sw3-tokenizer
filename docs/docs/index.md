# Start

**gpt-sw3-tokenizer** - Train, evaluate and analyze BPE tokenizers.

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

<div align="center">
<svg xmlns="http://www.w3.org/2000/svg" height="4em" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path d="M3.9 54.9C10.5 40.9 24.5 32 40 32H472c15.5 0 29.5 8.9 36.1 22.9s4.6 30.5-5.2 42.5L320 320.9V448c0 12.1-6.8 23.2-17.7 28.6s-23.8 4.3-33.5-3l-64-48c-8.1-6-12.8-15.5-12.8-25.6V320.9L9 97.3C-.7 85.4-2.8 68.8 3.9 54.9z"/></svg>
&nbsp;
&nbsp;
&nbsp;
<svg xmlns="http://www.w3.org/2000/svg" height="4em" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path d="M184 0c30.9 0 56 25.1 56 56V456c0 30.9-25.1 56-56 56c-28.9 0-52.7-21.9-55.7-50.1c-5.2 1.4-10.7 2.1-16.3 2.1c-35.3 0-64-28.7-64-64c0-7.4 1.3-14.6 3.6-21.2C21.4 367.4 0 338.2 0 304c0-31.9 18.7-59.5 45.8-72.3C37.1 220.8 32 207 32 192c0-30.7 21.6-56.3 50.4-62.6C80.8 123.9 80 118 80 112c0-29.9 20.6-55.1 48.3-62.1C131.3 21.9 155.1 0 184 0zM328 0c28.9 0 52.6 21.9 55.7 49.9c27.8 7 48.3 32.1 48.3 62.1c0 6-.8 11.9-2.4 17.4c28.8 6.2 50.4 31.9 50.4 62.6c0 15-5.1 28.8-13.8 39.7C493.3 244.5 512 272.1 512 304c0 34.2-21.4 63.4-51.6 74.8c2.3 6.6 3.6 13.8 3.6 21.2c0 35.3-28.7 64-64 64c-5.6 0-11.1-.7-16.3-2.1c-3 28.2-26.8 50.1-55.7 50.1c-30.9 0-56-25.1-56-56V56c0-30.9 25.1-56 56-56z"/></svg>
&nbsp;
&nbsp;
&nbsp;
<svg xmlns="http://www.w3.org/2000/svg" height="4em" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path d="M177.9 494.1c-18.7 18.7-49.1 18.7-67.9 0L17.9 401.9c-18.7-18.7-18.7-49.1 0-67.9l50.7-50.7 48 48c6.2 6.2 16.4 6.2 22.6 0s6.2-16.4 0-22.6l-48-48 41.4-41.4 48 48c6.2 6.2 16.4 6.2 22.6 0s6.2-16.4 0-22.6l-48-48 41.4-41.4 48 48c6.2 6.2 16.4 6.2 22.6 0s6.2-16.4 0-22.6l-48-48 41.4-41.4 48 48c6.2 6.2 16.4 6.2 22.6 0s6.2-16.4 0-22.6l-48-48 50.7-50.7c18.7-18.7 49.1-18.7 67.9 0l92.1 92.1c18.7 18.7 18.7 49.1 0 67.9L177.9 494.1z"/></svg>
&nbsp;
&nbsp;
&nbsp;
<svg xmlns="http://www.w3.org/2000/svg" height="4em" viewBox="0 0 512 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path d="M416 208c0 45.9-14.9 88.3-40 122.7L502.6 457.4c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 376c-34.4 25.2-76.8 40-122.7 40C93.1 416 0 322.9 0 208S93.1 0 208 0S416 93.1 416 208zM208 352a144 144 0 1 0 0-288 144 144 0 1 0 0 288z"/></svg>
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
</div>

-----------
## Features

:fontawesome-solid-filter: [Sampling](sampling.md)

- customizable amount of (disjunct) sampled data for training and evaluation
- weighting of different categories and languages

:fontawesome-solid-brain: [Training](training.md)

- support for SentencePiece and HuggingFace
- customizable tokenizer features (vocabulary size, handling of whitespace and numbers, ..)

:fontawesome-solid-ruler: [Evaluation](evaluation.md) 

- computation of common tokenizer metrics (unknown rate, fertility, proportion of continued words, ..)

:fontawesome-solid-magnifying-glass: [Analysis](analysis.md)

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