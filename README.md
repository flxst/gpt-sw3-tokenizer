# gpt-sw3-tokenizer

Tokenizer for the GPT-SW3 project (multilingual, Nordic Pile)

## Repo Structure

- `./notebooks`: contains notebooks
- `./src`: contains source code
- `.`: contains python and bash scripts as well as environment variables (`env.ini`)



## Setup

- Create virtual environment

- Install dependencies:

        pip install -r requirements.txt

- Specify the following folders in `./env.ini`:
  - `<data_original>`: contains original text data
  - `<data_sampled>`: contains sampled text data
  - `<output>`: contains trained tokenizer (incl. vocabulary, merge rules, parameters)

## Main Usage

To train a tokenizer, 
- move your data to the `<data_sampled>` folder, e.g. `<data_sampled>/my-data-*.jsonl`
- choose a name for the resulting tokenizer, e.g. `tokenizer1`
- run the training:

  ```
  python script_train.py 
      --dataset_files <data_sampled>/my-data-1 <data_sampled>/my-data-2
      --dataset_name tokenizer1
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


## Advanced Usage

The repository contains scripts & notebooks that 
allow to experiment with different parameters 
and analyze the results.

### Data & Testing (Optional Preparations)

- ```
  python script_create_test_data.py
  ```

  creates the following data for testing: 
    - <data_sampled>/test.json   (contains TEST_CORPUS)
    - <data_sampled>/code.json   (contains script_train.py as string)
    - <data_sampled>/fibrec.json (contains fibRec function as string)

- ```
  python script_test_load_dataset.py
  --dataset_files <dataset_files>
  --batch_size <batch_size>
  ```
  - loads the data in <dataset_files> in batches of <batch_size>
  - uses the get_training_corpus generator to read it
  - prints information to check that everything works as expected

- ```
  python script_split_data.py  
   --dataset_file <dataset_file>
   --max_sentence_length <max_sentence_length>
  ```
  - splits the documents in <dataset_file> such that they contain <max_sentence_length> characters

### Training

- Train Tokenizer: 
  - see `Main Usage`
  - in addition to single runs, 
  the bash script `bash train.sh` allows to systematically 
  execute multiple runs for the purpose of experimentation.

- ```
  python script_upsampling.py
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
  python script_apply_tokenizer.py --id HHMMSS
  ```
  - loads the tokenizer with the given <id> (that needs to be present in the folder <output>/<id>_*)
  - applies it to the data in TEST_EXAMPLES and prints the result
  
- ```
  python script_evaluate.py
  [parameters = <tokenizers>, <datasets> are hardcoded in the script]
  ```
  - applies each tokenizer on each dataset and computes unk_rate & closeness_to_character_level
  - writes results to `<output>/evaluation/results_*.json`


- `notebooks/tokenizer_analysis.ipynb`
  - examine tokenized test examples (effect of parameters)
  - examine subword lengths & effect of min_frequency
  - vocab_size: creates plots
    - vocabulary overlap 
    - unk_rate & closeness_to_char_level

## REAL DATA

### 1. Data

- Have the (sampled)
  - training data
  - evaluation data

  ready in the folder `<data_sampled>`

### 2. Tokenizer Training

- In `DATA_TRAIN.sh`, specify the training datasets (for each language) and vocab_size

- Run `bash train.sh`

### 3. Evaluation
(unk_rate & closeness_to_character_level)

- In `DATA_EVALUATION.py`, specify the evaluation datasets and vocab_sizes

- Run `script_evaluate.py`

  (make sure that "get_info_automatically" part is used)

### 4. Analysis

- Move tokenizer folders `<output>/*v<VOCAB_SIZE>*` to `<output>/multilinguality`

  (not the one with a different vocab_size created in 3.)
- Run notebook `./notebooks/tokenizer_analysis.ipynb`
