# gpt-sw3-tokenizer

Tokenizer for the GPT-SW3 project (multilingual, Nordic Pile)

## Repo Structure

- `.`: contains main python and bash scripts as well as environment variables (`env.ini`)
- `./notebooks`: contains notebooks
- `./scripts`: contains helper and testing python scripts
- `./src`: contains source code



## Setup

- Create virtual environment

- Install dependencies:

        pip install -r requirements.txt

- Specify the following folders in `./env.ini`:
  - `<data_original>`: contains original text data
  - `<data_sampled>`: contains sampled text data for training
  - `<data_eval>`: contains sampled text data for evaluation
  - `<output>`: contains trained tokenizer (incl. vocabulary, merge rules, parameters)

## A. Main Usage

Training a tokenizer requires the following steps:
1. Sampling
2. Training
3. Evaluation
4. Analysis

### 1. Sampling

Often times (especially in the case of very large datasets), 
one only wants to use a certain fraction of the original data for the tokenizer training (and evaluation).

Note:
- #1: If you want to skip this step, just point the 
`<data_sampled>` to the `<data_original>` folder 
in `env.ini` and proceed with the training.


- #2: Sampling is only implemented for the case where you have
categories (e.g. books, articles, ..) and languages (e.g. sv, da, ..)
and your data files have the format `<data_original>/<category>_<language>.jsonl`

To sample data from the original files in `<data_original>`, do the following: 
- Specify the categories, languages and weights in `SAMPLING_WEIGHTS.csv`
- Choose the fraction of your samples in percent, e.g. `<percent> = 10`
- Run the following script:

  ```
  python script_sampling.py 
      --percent <percent>   # e.g. 10
      [--evaluation 0]      # 0 = <data_sampled>, 1 = <data_eval>
  ```

The sampled files are called `<category>_<language>_<percent>p.jsonl` and can be found at
- `<data_sampled>` if `--evaluation 0` is used
- `<data_eval>` if `--evaluation 1` is used

- and ready to be used for training & evaluation in the next steps.


### 2. Training

To train the tokenizer on data in the `<data_sampled>` folder, do the following: 
- choose a name for the resulting tokenizer, e.g. `<tokenizer_name> = tokenizer1`
- choose your data files in `<data_sampled>`, e.g. 
  - `<dataset_files> = all` (which uses all files) or 
  - `<dataset_files> = data-1.jsonl data-2.jsonl` 
    (which uses two specific files) 

- run the training:

  ```
  python script_train.py 
      --tokenizer_name <tokenizer_name>      # e.g. tokenizer1
      --dataset_files <dataset_files>        # e.g. "all" = all files in <data_sampled>
      [--dataset_filter all]                 # e.g. "all" = no filter
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

The trained tokenizer can be found e.g. in the folder `<output>/*_tokenizer1`. 
In particular, the model is named
- `model.model` (if library == "SP")
- `tokenizer.json` (if library == "HF")

In addition, there are two files that (together with the data) are to be used with 
Megatron-LM's data preprocessing tool (`tools/preprocess_data.py`)
- `tokenizer_vocab.json` (vocabulary)
- `tokenizer_merge.txt` (merge rules)

Note: In case library == "SP" is used, the `tokenizer_merge.txt` file is missing. See "Advanced Usage" for more details.

### 3. Evaluation

To evaluate the tokenizer on data in the `<data_eval>` folder, do the following: 
- choose the name of the tokenizer, e.g. `<tokenizer_name> = tokenizer1`
- choose the vocab size of the tokenizer, e.g. `<vocab_sizes> = 64000`
- run the evaluation:

  ```
  python script_evaluate.py 
      --tokenizer_name <tokenizer_name>  # e.g. tokenizer1 
      --vocab_sizes <vocab_sizes>        # e.g. 64000
  ```

This
- applies the tokenizer `<output>/*_<tokenizer_name>` on each dataset `<dataset_eval>` in `<data_eval>`
- computes evaluation metrics (unk rate, ctcl, fertility, proportion of continued words, token frequencies)
- writes results to 
  - `<output>/evaluation/results_<tokenizer_name>.json`
  - `<output>/evaluation/token_frequencies_<tokenizer_name>_<dataset_eval>.json`

### 4. Analysis

To analyze the trained tokenizers (and inspect the evaluation metrics), do the following: 
- run the following notebook:

  ```
  notebooks/tokenizer_analysis.ipynb
  ```

This allows to examine
- tokenized test examples (effect of parameters)
- subword lengths & effect of min_frequency
- evaluation metrics


## B. Experiments (Vocabulary & Languages)

For experiments with different 
- vocabulary sizes and
- multiple languages

only a few things need to be adjusted with respect to "Main Usage"

### 1. Sampling
exactly like "Main Usage"

### 2. Tokenizer Training

- In `train.sh`, specify the
  - tokenizer names (6 single-language and 1 multi-language dataset)
  - their training datasets (6 single-language and 1 multi-language dataset)
  - their (maximum) vocab size `<vocab_size>` (e.g. `<vocab_size> = 128000`)

- Run `bash train.sh`

### 3. Evaluation

- For the multi-language tokenizer you want to evaluate, take 
  - its name (e.g. `<tokenizer_name> = 4all-a1.0`) and 
  - its (maximum) vocab size (e.g. `<vocab_size> = 128000`) and 
  - add pruned vocab sizes you want to test (e.g. `<vocab_sizes> = 64000 96000 128000`)


- Run 
  ```
  python script_evaluate.py 
      --tokenizer_name <tokenizer_name>  # e.g. 4all-a1.0 
      --vocab_sizes <vocab_sizes>        # e.g. 64000 96000 128000
  ```

### 4. Analysis

To analyze the multi-language tokenizer (evaluation, pruned versions) and compare it with the single-language tokenizers (vocabulary overlap), 
do the following

- Preparation:
  - Take the `<tokenizer_number>` that `<tokenizer_name>` starts with, e.g. `<tokenizer_number> = 1` (specified in `train.sh`)
  - Take the (maximum) `<vocab_size>` (used in 2./3.)
  - Run
    ```
    python scripts/analysis_helpers/script_move_experimental_tokenizers.py 
        --tokenizer_number <tokenizer_number>  # e.g. 4
        --vocab_size <vocab_size>              # e.g. 128000
    ```
    
  - This moves the tokenizer folders `<output>/*v<vocab_size>_<tokenizer_number><language>*` to `<output>/multilinguality`

    (not the ones with pruned vocab sizes created in 3.)


- Run the following notebook:

  ```
  notebooks/tokenizer_analysis.ipynb
  ```

  - exactly like "Main Usage"
  - notebook analyses vocabulary overlap in addition


## C. Advanced Usage

The repository contains additional scripts that 
provide extended functionalities and allow for testing.

### Testing

- ```
  python scripts/tests/script_test_load_dataset.py
  --dataset_files <dataset_files>
  --batch_size <batch_size>
  ```
  - loads the data in <dataset_files> in batches of <batch_size>
  - uses the get_training_corpus generator to read it
  - prints information to check that everything works as expected


### Data (Optional Preparations)

- ```
  python scripts/create_test_data/script_create_test_data_sampled.py
  ```

  creates the following data for testing: 
    - <data_sampled>/test.json   (contains TEST_CORPUS)
    - <data_sampled>/code.json   (contains script_train.py as string)
    - <data_sampled>/fibrec.json (contains fibRec function as string)

- ```
  python scripts/data_helpers/script_split_data.py  
   --dataset_file <dataset_file>
   --max_sentence_length <max_sentence_length>
  ```
  - splits the documents in <dataset_file> such that they contain <max_sentence_length> characters

### Upsampling

- ```
  python scripts/upsampling/script_upsampling.py
  --dataset_files <dataset_files>
  --stats <stats>
  --total <total>
  --alpha <alpha>  # upsampling parameter, 0 <= alpha <= 1
  ```
  - computes the upsampling factors for each dataset, using alpha parameter
  - write upsampling factors to `data/file-upsampled.json`
      
  - in addition to single runs,
  the bash script `bash upsampling.sh` allows to systematically
  execute multiple runs for the purpose of experimentation.

### Analysis

- ```
  python scripts/application_helpers/script_apply_tokenizer.py --id HHMMSS
  ```
  - loads the tokenizer with the given <id> (that needs to be present in the folder <output>/<id>_*)
  - applies it to the data in TEST_EXAMPLES and prints the result

### Conversion of tokenizer from SentencePiece to HuggingFace (incomplete)

- ```
  python scripts/application_helpers/script_merge.py
  ```
  - loads a <tokenizer_vocab> file (hardcoded in script)
  - writes a <tokenizer_merge> file
  
- ```
  python scripts/application_helpers/script_test_conversion_from_sp_to_hf.py
  ```
  - loads a SP tokenizer from a <model_file> (hardcoded)
  - loads the corresponding HF tokenizer from a <tokenizer_vocab> file and a <tokenizer_merge> file
    (Note: the <tokenizer_merge> file can be created by script_merge.py)
  - compares the two tokenizers (vocab & examples)
