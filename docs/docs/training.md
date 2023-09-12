# :fontawesome-solid-brain: Training

---
## How-To
To train a tokenizer on data in the `<data_train>` folder, run `script_train.py`:


???+ note "train tokenizer"
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
      [--train_extremely_large_corpus 1]     # 0, 1
    ```

**Arguments:**

- `--tokenizer_name` will be the name of the tokenizer, e.g. `tokenizer1`
- `--dataset_files` specifies the data files in `<data_train>` you would like to include, e.g.
    - `<dataset_files> = all` (which uses all files) or
    - `<dataset_files> = data_en.jsonl data_sv.jsonl`
       (which uses two specific files)

<br>
**Optional Arguments:**

- `--dataset_filter` is an alternative to `--dataset_files` (which needs to be set to `all`). Any files that contain the specified substring will be included
- `--monolingual` trains (multiple) monolingual tokenizers instead of a single multilingual one
- `--library` specifies the library to use (`SP` = SentencePiece or `HF` = HuggingFace)
- `--unicode_normalization` specifies the unicode normalization that the data is preprocessed with
- `--individual_digits` splits digits into separate tokens using whitespace
- `--add_whitespace_tokens` adds 23 consecutive whitespace tokens to the tokenizer's vocabulary
- `--add_code_tokens` adds special code tokens to the tokenizer's vocabulary. These are specified in `CODE_TOKENS.csv`
- `--minimum_frequency` specifies the minimum frequency required for a token to be added to the vocabulary
   Note that this most likely results in a vocabulary size smaller than the vocabulary size specified beforehand
- `--vocab_size` specifies the desired vocabulary size

<br>
**Optional Arguments only available for SentencePiece:**

- `--add_prefix_space`
- `--byte_fallback`
- `--character_coverage`
- `--train_extremely_large_corpus`

The corresponding SentencePiece features are used. We refer to the [SentencePiece documentation](https://github.com/google/sentencepiece/blob/master/doc/options.md) for further details.


---
## Results

The trained tokenizer is saved at the folder `<output>/YYmmdd_HHMMSS-v<vocab_size>_<tokenizer_name>`
and contains the following files:

- Tokenizer: `model.model` & `model.vocab` (if `library == "SP"`) or tokenizer.json (if `library == "HF"`)

    ???+ example "model.vocab (SentencePiece)"
        ```
        <pad> 0
        <unk> 0
        <s> 0
        <|endoftext|> 0
        <|javascript|> 0
        <|python|> 0
        <|sql|>	0
        <|shell|> 0
        <0x00> 0
        <0x01> 0

        [..]

        <0xFE> 0
        <0xFF> 0
        ▁t -0
        he -1
        ▁a -2

        [..]

        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ -63713
        ```

    ???+ example "tokenizer.json (HuggingFace)"
        ```
        {
            [..]
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<|javascript|>",
                    [..]
                    "special": true
                },
                {
                    "id": 1,
                    "content": "<|python|>",
                    [..]
                    "special": true
                },
                {
                    "id": 2,
                    "content": "<|sql|>",
                    [..]
                    "special": true
                },
                {
                    "id": 3,
                    "content": "<|shell|>",
                    [..]
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": true,
                        "trim_offsets": true,
                        "use_regex": true
                    },
                    {
                        "type": "Digits",
                        "individual_digits": true
                    }
                ]
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false,
                "use_regex": true
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "vocab": {
                "<|javascript|>": 0,
                "<|python|>": 1,
                "<|sql|>": 2,
                "<|shell|>": 3,
                "!": 4,
                "\"": 5,

                [..]

                "e on",
                "e res",
                "e ase"
            }
        }
        ```


- Training parameters: `parameters.txt`

    ???+ example "parameters.txt"
        ```
        library = SP
        dataset_files = [..]
        tokenizer_name = tokenizer1
        unicode_normalization = None
        individual_digits = True
        add_prefix_space = True
        add_whitespace_tokens = 2
        add_code_tokens = 1
        minimum_frequency = 0
        byte_fallback = True
        character_coverage = 0.9999
        train_extremely_large_corpus = True
        vocab_size = 63977
        vocab_size_external = 64000
        special_tokens = ['<|javascript|>', '<|python|>', '<|sql|>', '<|shell|>']
        timestamp = 230912_110630
        output_dir = [..]
        ```

- Tokenizer Vocabulary Statistics: `tokenizer_subword_lengths.json`

    ???+ example "tokenizer_subword_lengths.json"
        ```
        {
            "1": 173,
            "2": 1700, 
            "3": 4965, 
            "4": 8477, 
            [..]
            "24": 1, 
            "mean": 6.656015625, 
            "vocab_size": 64000,
        }
        ```

- Dataset Statistics: `overview.json`

    ???+ example "overview.json"
        ```
        {
            "files": 1, 
            "documents_total": 8625, 
            "documents": 8625, 
            "dataset_files": [..], 
            "data_size_total": "0.0121G", 
            "data_size": ["0.0032G", "0.0022G", "0.0014G", "0.0004G", "0.0000G", "0.0038G", "0.0011G"], 
            "time": "7.95s",
        }
        ```
