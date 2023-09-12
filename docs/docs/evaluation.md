# :fontawesome-solid-ruler: Evaluation

---
## How-To
To evaluate the tokenizer on data in the `<data_eval>` folder, run `script_evaluate.py`:

???+ note "evaluate tokenizer"
    ```
    python script_evaluate.py 
      --tokenizer_name <tokenizer_name>         # e.g. tokenizer1 
      [--vocab_size <vocab_size>]               # e.g. 64000
      [--vocab_size_pruned <vocab_size_pruned>] # e.g. 40000 51200
      [--monolingual]                           # if used, monolingual models are evaluated
    ```

**Arguments:**

- `--tokenizer_name` is the name of the tokenizer, e.g. `tokenizer1`

<br>
**Optional Arguments:**

- `--vocab_size` specifies the vocabulary size of the tokenizer 
   (useful if there are several tokenizers with the same `tokenizer_name` but different vocabulary sizes)
- `--vocab_size_pruned` evaluates the tokenizer with different, pruned vocabulary sizes 
- `--monolingual` evaluates (multiple) monolingual tokenizers (trained with the same flag)

<br>
The script

- applies the tokenizer `<output>/*_<tokenizer_name>` on each dataset `<dataset_eval>` in `<data_eval>`
- computes the following evaluation metrics (see the [paper](https://arxiv.org/abs/2304.14780) for details):

    - `unknown_rate`
    - `fertility`
    - `proportion_of_continued_words`
    - `token_frequencies`

---
## Results

Results are written to 

- `<output>/evaluation/results_<tokenizer_name>.json`
- `<output>/evaluation/token_frequencies_<tokenizer_name>.json`

They will be used for the [analysis](analysis.md).
