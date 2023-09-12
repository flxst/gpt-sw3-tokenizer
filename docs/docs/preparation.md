# :fontawesome-solid-circle-check: Preparation

---
## Environment

In the environment file `./env.ini`, one needs to specify paths for the following folders:

- `<data_original>`: contains original text data
- `<data_train>`: contains sampled text data for training
- `<data_eval>`: contains sampled text data for evaluation
- `<output>`: contains trained tokenizers (incl. vocabulary, merge rules, parameters)

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
