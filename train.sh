# INPUT=data/test.json
# DATASET="0"
# VOCAB_SIZE=100

# INPUT=data/books_sv_epub.jsonl
# DATASET="1"
# VOCAB_SIZE=50000

# INPUT=data/anforanden.jsonl
# DATASET="2"
# VOCAB_SIZE=30000

# INPUT="data/anforanden.jsonl data/code.json"
# DATASET="2p"
# VOCAB_SIZE=100000

INPUT="data/wiki_da.jsonl"
DATASET="3da"
# INPUT="data/wiki_en.jsonl"
# DATASET="3en"
# INPUT="data/wiki_is.jsonl"
# DATASET="3is"
# INPUT="data/wiki_no.jsonl"
# DATASET="3no"
# INPUT="data/wiki_sv.jsonl"
# DATASET="3sv"
INPUT="data/wiki_da.jsonl data/wiki_en.jsonl data/wiki_is.jsonl data/wiki_no.jsonl data/wiki_sv.jsonl"
DATASET="3all"
# INPUT="data/wiki_da.jsonl data/wiki_en.jsonl data/wiki_is.jsonl data/wiki_no.jsonl data/wiki_sv.jsonl data/code.json"
# DATASET="3all+"
VOCAB_SIZE=100000

### EXCLUDED OPTIONS ####
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 0 --individual_digits 0 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 0 --individual_digits 0 --minimum_frequency 0 --unicode_normalization NFKD --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 0 --individual_digits 0 --minimum_frequency 0 --unicode_normalization NFKC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 0 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0

### TEST OPTIONS ###
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 0 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 0 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 10

### CURRENT OPTIONS ####
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 2 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 4 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 6 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 8 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 10 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 12 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 14 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 16 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 18 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
python script_train_tokenizer.py --input $INPUT --dataset $DATASET --add_prefix_space 1 --individual_digits 1 --minimum_frequency 20 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 24
