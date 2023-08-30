## Advanced Usage

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
    - `<data_train>/test.json`   (contains TEST_CORPUS)
    - `<data_train>/code.json`   (contains script_train.py as string)
    - `<data_train>/fibrec.json` (contains fibRec function as string)

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
