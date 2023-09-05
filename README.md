# gpt-sw3-tokenizer

Train, evaluate and analyze BPE tokenizers.

**Features:**
- data sampling 
- data weighting of subsets with different categories and languages
- training with SentencePiece or HuggingFace
- customizable tokenizer features (vocabulary size, handling of whitespace and numbers, ..)
- detailed evaluation and analysis (e.g. computation of common tokenizer metrics, example tokenization, vocabulary and performance comparison across languages, effect of the vocabulary size, ..)

This repository was used to train a multilingual, BPE-based SentencePiece tokenizer for the [GPT-SW3](https://arxiv.org/abs/2305.12987) model family on the Nordic Pile dataset.
See [this paper](https://arxiv.org/abs/2304.14780) for more details.

----
## Repository Structure

- `.`: contains main python and bash scripts as well as settings (`env.ini` and `SAMPLING_WEIGHTS.csv`)
- `./notebooks`: contains notebooks
- `./scripts`: contains helper and test python scripts
- `./src`: contains source code

## Setup

- Create virtual environment

- Install dependencies:

        pip install -r requirements.txt

- Specify the following folders in `./env.ini`:
  - `<data_original>`: contains original text data
  - `<data_train>`: contains sampled text data for training
  - `<data_eval>`: contains sampled text data for evaluation
  - `<output>`: contains trained tokenizer (incl. vocabulary, merge rules, parameters)


- Data format

    - The files contained in the folder `<data_original>` must adhere to the following naming convention:
      ```
      <category>_<language>.jsonl
      ```
      Data split into multiple categories (e.g. books, articles, ..) and/or languages (e.g. sv, da, ..)
      like this can be weighted in a customized way (as described below).

    - Each row in a file needs to be formatted like this:

      ```
      {"text": "..."} 
      ```
      Fields other than `"text"` may be present but will be ignored. 


----
## Usage

Training a tokenizer requires the following steps:
1. Sampling
2. Training
3. Evaluation
4. Analysis

### 1. Sampling

*Note: If you want to skip this step, just point the
`<data_train>` to the `<data_original>` folder
in `env.ini` and proceed with the training.*

Often times, especially in the case of very large datasets, 
one only wants to use a certain fraction of the original data for the tokenizer training (and evaluation).
In addition, the data is to be weighted for tokenizer (and model) training, see 
e.g. [GPT-3](https://arxiv.org/abs/2005.14165) or [GPT-SW3](https://arxiv.org/abs/2305.12987).

To sample (and weight) data from the original files in `<data_original>`, do the following: 
- Specify the categories, languages and weights in `SAMPLING_WEIGHTS.csv`
- Choose the fraction of your samples in percent, e.g. `<percent> = 10`
- Run the following script:

  ```
  python script_sampling.py 
      --percent <percent>   # e.g. 10
      [--evaluation 0]      # 0 = <data_train>, 1 = <data_eval>
  ```

The sampled (and weighted) data files are called `<category>_<language>_<percent>p.jsonl` and can be found in the folder
- `<data_train>` if `--evaluation 0` is used
- `<data_eval>` if `--evaluation 1` is used

In the next steps, they are used for training and evaluation, respectively.


### 2. Training

To train the tokenizer on data in the `<data_train>` folder, do the following: 
- choose a name for the resulting tokenizer, e.g. `<tokenizer_name> = tokenizer1`
- choose your data files in `<data_train>`, e.g. 
  - `<dataset_files> = all` (which uses all files) or 
  - `<dataset_files> = data-1.jsonl data-2.jsonl` 
    (which uses two specific files) 

- run the training:

  ```
  python script_train.py 
      --tokenizer_name <tokenizer_name>      # e.g. tokenizer1
      --dataset_files <dataset_files>        # e.g. "all" = all files in <data_train>
      [--dataset_filter all]                 # e.g. "all" = no filter
      [--monolingual]                        # if used, monolingual models are trained
      [--library SP]                         # SP = SentencePiece, HF = HuggingFace
      [--unicode_normalization None]         # None, NFC, NFKC
      [--individual_digits 1]                # 0, 1
      [--add_prefix_space 1]                 # 0, 1
      [--add_whitespace_tokens 2]            # 0, 1 (added at top of vocab), 2 (added at bottom of vocab)
      [--add_code_tokens 1]                  # 0, 1 (added at top of vocab)
      [--minimum_frequency 0]                # int >= 0
      [--byte_fallback 1]                    # 0, 1
      [--character_coverage 0.9999]          # float, useful if byte_fallback = 1
      [--vocab_size 64000]                   # int, divisible by 128
      [--alpha -1]                           # upsampling parameter, -1 or 0 <= alpha < 1
      [--train_extremely_large_corpus 1]     # 0, 1
  ```

The trained tokenizer is saved at `<output>/YYmmdd_HHMMSS-v<vocab_size>_<tokenizer_name>`
and contains the following files:
- Tokenizer: `model.model` & `model.vocab` (if library == "SP") or tokenizer.json (if library == "HF")
- Training parameters: `parameters.txt`
- Files for Megatron-LM: `tokenizer_vocab.json` & `tokenizer_merge.txt`
- Tokenizer Statistics: `tokenizer_subword_lengths.json`
- Dataset Statistics: `overview.json`

**Advanced Usage:**
- To train (multiple) monolingual tokenizers, use `--monolingual`


### 3. Evaluation

To evaluate the tokenizer on data in the `<data_eval>` folder, do the following: 
- choose the name of the tokenizer, e.g. `<tokenizer_name> = tokenizer1`
- choose the vocab size of the tokenizer, e.g. `<vocab_sizes> = 64000`
- run the evaluation:

  ```
  python script_evaluate.py 
      --tokenizer_name <tokenizer_name>         # e.g. tokenizer1 
      [--vocab_size <vocab_size>]               # e.g. 64000
      [--vocab_size_pruned <vocab_size_pruned>] # e.g. 40000 51200
      [--monolingual]                           # if used, monolingual models are evaluated
  ```

This
- applies the tokenizer `<output>/*_<tokenizer_name>` on each dataset `<dataset_eval>` in `<data_eval>`
- computes evaluation metrics (unk rate, ctcl, fertility, proportion of continued words, token frequencies)
- writes results to 
  - `<output>/evaluation/results_<tokenizer_name>.json`
  - `<output>/evaluation/token_frequencies_<tokenizer_name>.json`

**Advanced Usage:** 
- If `vocab_size_pruned` is specified, variants of the tokenizer with pruned vocabularies are evaluated in addition.
- To evaluate (multiple) monolingual tokenizers, use `--monolingual`

### 4. Analysis

To analyze the trained tokenizers (and inspect the evaluation metrics), run the following notebook:

  ```
  notebooks/tokenizer_analysis.ipynb
  ```

This allows to examine e.g.
- tokenized examples
- evaluation metrics
- vocabulary and performance comparison across languages
- effect of the vocabulary size