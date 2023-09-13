# :fontawesome-solid-filter: Sampling

*Note: If you want to skip this step, just point the
`<data_train>` to the `<data_original>` folder
in the [environment file](preparation.md#environment) and proceed with the [training](training.md).*

Often times, especially in the case of very large datasets,
one only wants to use a certain fraction of the original data for the tokenizer training (and evaluation).
In addition, the data is to be weighted for tokenizer (and model) training, see
e.g. [GPT-3](https://arxiv.org/abs/2005.14165) or [GPT-SW3](https://arxiv.org/abs/2305.12987).

---
## How-To

To sample (and weight) data from the original files in `<data_original>`, take the following steps:

- Specify the categories, languages and their corresponding weights in `SAMPLING_WEIGHTS.csv`:

    ???+ example "SAMPLING_WEIGHTS.csv"
        ```
        category,sv,en
        articles,1,0.5
        books,0.7,1
        ```

    The above example contains 2 categories (`articles` & `books`) and 2 languages (`sv` & `en`).
  
- Run `script_sampling.py`:

    ???+ note "sample data"
        ```
        python script_sampling.py 
            --percent <percent>   # e.g. 10
            [--evaluation 0]      # 0 = <data_train>, 1 = <data_eval>
        ```
    **Arguments:**

    - `--percent` is the fraction of documents with respect to original data in percent

    <br>
    **Optional Arguments:**

    - `--evaluation` can be used to sample data for evaluation instead of training

Note that for each combination $x = cl$ of a category $c$ and language $l$, the fraction of sampled documents is given by the product of the
individual weight $W_x$ read from `SAMPLING_WEIGHTS.csv` and the global factor $p$ specified via `--percent`:

$$
\left(\frac{{\text{number of sampled documents}}}{{\text{number of original documents}}}\right)_x = W_x \cdot p 
$$

---
## Results

The sampled (and weighted) data files are called `<category>_<language>.jsonl` (just like their original counterparts, see [preparation](preparation.md#data-format)) and can be found in the folder

- `<data_train>` if `--evaluation 0` is used
- `<data_eval>` if `--evaluation 1` is used

In addition, the folder contains a log file `SAMPLING.log` which contains information about the sampling process.

In the next steps, the data are used for [training](training.md) and [evaluation](evaluation.md), respectively.
