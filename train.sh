# DATASET_FILES=data/test.json
# DATASET_NAME="0"
# VOCAB_SIZE=100

# DATASET_FILES=data/books_sv_epub.jsonl
# DATASET_NAME="1"
# VOCAB_SIZE=50000

# DATASET_FILES=data/anforanden.jsonl
# DATASET_NAME="2"
# VOCAB_SIZE=30000

# DATASET_FILES="data/anforanden.jsonl data/code.json"
# DATASET_NAME="2p"
# VOCAB_SIZE=100000

##################################################################################################
### WIKI: SINGLE LANGUAGE ##############################################################################
##################################################################################################
# DATASET_FILES="data/wiki_da.jsonl"
# DATASET_NAME="3da"
# DATASET_FILES="data/wiki_en.jsonl"
# DATASET_NAME="3en"
DATASET_FILES="data/wiki_is.jsonl"
DATASET_NAME="3is"
# DATASET_FILES="data/wiki_no.jsonl"
# DATASET_NAME="3no"
# DATASET_FILES="data/wiki_sv.jsonl"
# DATASET_NAME="3sv"
VOCAB_SIZE=250000
ALPHA=-1

##################################################################################################
### WIKI: ALL ##########################################################################################
##################################################################################################
# DATASET_FILES="data/wiki_da.jsonl data/wiki_en.jsonl data/wiki_is.jsonl data/wiki_no.jsonl data/wiki_sv.jsonl"
# ALPHA=1.0

# DATASET_FILES="data/wiki_da.jsonl data/wiki_da.jsonl
#  data/wiki_en.jsonl
#  data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl
#  data/wiki_no.jsonl data/wiki_no.jsonl
#  data/wiki_sv.jsonl data/wiki_sv.jsonl"
# ALPHA=0.8

# DATASET_FILES="data/wiki_da.jsonl data/wiki_da.jsonl data/wiki_da.jsonl data/wiki_da.jsonl
#  data/wiki_en.jsonl
#  data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl
#  data/wiki_no.jsonl data/wiki_no.jsonl data/wiki_no.jsonl data/wiki_no.jsonl
#  data/wiki_sv.jsonl data/wiki_sv.jsonl data/wiki_sv.jsonl"
# ALPHA=0.6

# DATASET_NAME="3all"
# VOCAB_SIZE=250000

##################################################################################################
### FINAL100: SINGLE LANGUAGE ####################################################################
##################################################################################################
# DATASET_FILES="data/data_100_final/books/final_da.jsonl
#  data/data_100_final/conversational/final_da.jsonl
#  data/data_100_final/misc/final_da.jsonl
#  data/data_100_final/web_commoncrawl/final_da.jsonl
#  data/data_100_final/web_sources/final_da.jsonl"
# DATASET_NAME="4da"
# DATASET_FILES="data/wiki_en.jsonl"
# DATASET_NAME="3en"
# DATASET_FILES="data/wiki_is.jsonl"
# DATASET_NAME="3is"
# DATASET_FILES="data/wiki_no.jsonl"
# DATASET_NAME="3no"
# DATASET_FILES="data/wiki_sv.jsonl"
# DATASET_NAME="3sv"

# VOCAB_SIZE=250000
# ALPHA=-1

##################################################################################################
### ALL+ #########################################################################################
##################################################################################################
# DATASET_FILES="data/wiki_da.jsonl data/wiki_en.jsonl data/wiki_is.jsonl data/wiki_no.jsonl data/wiki_sv.jsonl data/code.json"
# DATASET_NAME="3all+"
# VOCAB_SIZE=100000

### EXCLUDED OPTIONS ####
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 0 --individual_digits 0 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 0 --individual_digits 0 --minimum_frequency 0 --unicode_normalization NFKD --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 0 --individual_digits 0 --minimum_frequency 0 --unicode_normalization NFKC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 0 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 1 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0

### TEST OPTIONS ###
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 0 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 0
# python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --add_prefix_space 0 --individual_digits 1 --minimum_frequency 0 --unicode_normalization NFC --vocab_size $VOCAB_SIZE --add_whitespace_tokens 10

### CURRENT OPTIONS ####
python script_train.py --dataset_files $DATASET_FILES --dataset_name $DATASET_NAME --unicode_normalization NFC --individual_digits 1 --add_prefix_space 1 --add_whitespace_tokens 24 --minimum_frequency 0 --vocab_size $VOCAB_SIZE --alpha $ALPHA
