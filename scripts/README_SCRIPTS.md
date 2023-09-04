# Scripts for Advanced Usage

The repository contains additional scripts that
provide extended functionalities and allow for testing.

1. Create Test Data
2. Data Processing
3. Tokenizer Processing
4. Tokenizer Application

----
## 1. Create Test Data

### Original Test Data

- ```
  python script_create_test_data_original.py
  [--number_of_documents 20]
  ```

  for each combination of `<category>` & `<language>` (as specified in `SAMPLING_WEIGHTS.csv`), the script
  - creates `<number_of_documents>` fake original documents for testing
  - writes the fake original data to `<data_original>/<category>_<language>.jsonl`

### Sampled Test Data

- ```
  python script_create_test_data_sampled.py
  ```

  creates the following data for testing:
  - `<data_train>/test.json`
  - `<data_train>/code.json`
  - `<data_train>/fibrec.json`

----
## 2. Data Processing

### Load Dataset

- ```
  python script_test_load_dataset.py
  --dataset_files <dataset_files>
  --batch_size <batch_size>
  ```
  - loads the data in `<data_original>/<dataset_files>` in batches of `<batch_size>`
  - prints information to check that everything works as expected

### Apply Word Length Filter

- ```
  python script_apply_word_length_filter.py
  --directory <directory>
  [--threshold 20000]
  ```
  - takes all the dataset files in <directory>
  - for each dataset file, checks whether a document has a non-whitespace sequence of length > `<threshold>`
  - if so, it filters those documents and writes the rest to `<directory>_FILTERED`

### Change File Names

- ```
  python script_change_file_names.py
  --directory <directory>
  [--remove_percent 50]
  [--add_percent 50]
  ```
  - removes or adds the suffix `_{percent}p` for all the dataset files in `<directory>`

### Concatenate Data by Language

- ```
  python script_concatenate_data_by_language.py
  --directory <directory>
  ```
  - takes all the dataset files in <directory>
  - merges them by language
  - writes the results to <directory>_CONCATENATED_BY_LANGUAGE


### Split Data

- ```
  python script_split_data.py  
   --dataset_file <dataset_file>
   --max_sentence_length <max_sentence_length>
  ```
  - splits the documents in `<data_train>/<dataset_file>` such that they contain `<max_sentence_length>` characters

----
## 3. Tokenizer Processing

### Load Tokenizer SP
- ```
  python script_load_tokenizer_sp.py
  --tokenizer_directory 152808-v64000_tokenizer1
  [--verbose]
  ```
  - loads the tokenizer from `<output>/<tokenizer_directory>/model.model`
  - prints some infos

### Add Special Token SP
- ```
  python script_add_special_tokens_sp.py
  --tokenizer_directory <tokenizer_directory>
  ```
  - loads the tokenizer from `<output>/<tokenizer_directory>/model.model`
  - adds a special token
  - writes the new tokenizer to `<output>/<tokenizer_directory>___MST/model.model`

### Extract Vocabulary SP
- ```
  python script_extract_vocabulary_sp.py
  --tokenizer_directory <tokenizer_directory>
  ```
  - loads the tokenizer from `<output>/<tokenizer_directory>/model.model`
  - extracts the vocabulary and writes it to `<output>/<tokenizer_directory>/tokenizer_vocab.json`
  - the same is done by `script_evaluate.py`

### Create Merge File SP [unfinished]
- ```
  python script_create_merge_file_sp.py
  --tokenizer_directory 152808-v64000_tokenizer1
  ```
  - loads the tokenizer from `<output>/<tokenizer_directory>/model.model`
  - writes a merge file to `<output>/<tokenizer_directory>/tokenizer_merge.txt`


### Compare SP and HF [unfinished]

- ```
  python script_compare_sp_and_hf.py
  --tokenizer_directory <tokenizer_directory>
  ```
  - loads the SP tokenizer from `<output>/<tokenizer_directory>/model.model`
  - loads the corresponding HF tokenizer from `<output>/<tokenizer_directory>/tokenizer_vocab.json` and `[..]/tokenizer_merge.txt`
    (Note: the latter can be created by `script_create_merge_file_sp.py`)
  - compares the two tokenizers (vocab & examples)

----
## 4. Tokenizer Application

- ```
  python script_apply_tokenizer.py
  --tokenizer_directory <tokenizer_directory>
  [--SP]
  [--HF]
  ```
  - loads the tokenizer from `<output>/<tokenizer_directory>/tokenizer.json` (if `--HF`) or `[..]/model.model` (if `--SP`)
  - applies it to the data in TEST_EXAMPLES and prints the result
