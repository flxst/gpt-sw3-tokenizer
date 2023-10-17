# :fontawesome-solid-circle-check: Preparation

---
## Environment

The environment file `env.ini` can be used to specify paths and settings.
By default, it looks like this:

???+ example "env.ini"
    ```
    [main]
    data_original = data_original
    data_train = data_train
    data_eval = data_eval
    output = output

    [sampling]
    weights = SAMPLING_WEIGHTS.csv

    [other]
    debug = 0
    verbose = 0
    ```

In the `[main]` section, the following folders are given: 

- `<data_original>`: contains original text data
- `<data_train>`: contains sampled text data for training
- `<data_eval>`: contains sampled text data for evaluation
- `<output>`: contains trained tokenizers (incl. vocabulary, merge rules, parameters)

The `[sampling]` section contains the path to the sampling weights file (see [Sampling](sampling.md)).
The parameters in the `[other]` section should be 0 or 1 and can be used for debugging or verbose output. 

---
## Data Format

- The files contained in the folder `<data_original>` must adhere to the following naming convention:
  ```
  <category>_<language>.jsonl
  ```
  Data split into multiple categories (e.g. `books`, `articles`, ..) and/or languages (e.g. `sv`, `da`, ..)
  like this can be weighted in a customized way (as described in [Sampling](sampling.md)).

- Each row in a file needs to be formatted like this:
  ```
  {"text": "..."} 
  ```
  Fields other than `"text"` may be present but will be ignored. 
