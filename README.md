# gpt-sw3-tokenizer

Tokenizer for the GPT-SW3 project (multilingual, Nordic Pile)

## Setup

- Create virtual environment

- Install dependencies:

        pip install -r requirements.txt


## Usage

- Create Test Data: `python script_create_test_data.py`
- Train Tokenizer: 
  - single run: `python script_train_tokenizer.py [--input data/test_data.json]` (uses `parameters.py`)
  - all runs: `bash train.sh`
- Compute Upsampling:
  - single run: `python script_upsampling.py --input $INPUT --stats $STATS --total $TOTAL --alpha 0.6`
  - all runs: `bash upsampling.sh`
- Test Tokenizer: `python script_test_tokenizer.py --id HHMMSS`
- Analysis: `notebooks/tokenizer_analysis.ipynb`



## Structure

- `data`: contains text data
- `output`: contains trained tokenizer (incl. vocabulary, merge rules, parameters) 


