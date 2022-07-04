# gpt-sw3-tokenizer

Tokenizer for the GPT-SW3 project (multilingual, Nordic Pile)

## Setup

- Create virtual environment

- Install dependencies:

        pip install -r requirements.txt

## Repo Structure

- `data`: contains text data
- `notebooks`: contains notebooks
- `output`: contains trained tokenizer (incl. vocabulary, merge rules, parameters)
- `src`: contains source code

## Usage

### Optional Preparations (Data & Testing)

- [Optional] Create Test Data: `python script_create_test_data.py`
- [Optional] Load Dataset Testing: `python script_test_load_dataset.py`

### Training

- Train Tokenizer: 
  - single run: `python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME [..]`
  - all runs: `bash train.sh`

- [Optional] Compute Upsampling:
  - single run: `python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.6`
  - all runs: `bash upsampling.sh`

### Analysis

- Apply Tokenizer: `python script_apply_tokenizer.py --id HHMMSS`
- Analysis: `notebooks/tokenizer_analysis.ipynb`

### Next Steps

Once you have decided on a tokenizer (`output/HHMMSS_*`), take
- the vocabulary (`tokenizer_vocab.json`) and merge (`tokenizer_merge.txt`) files
- the data you used to train the tokenizer (in `data`)

and use it together with Megatron-LM's data preprocessing tool (`tools/preprocess_data.py`)
