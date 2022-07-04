#!/bin/bash

### STANDARD PARAMETERS ##############################
# LIBRARY="HF"
LIBRARY="SP"
UNICODE_NORMALIZATION="None"
INDIVIDUAL_DIGITS=1
ADD_PREFIX_SPACE=1
ADD_WHITESPACE_TOKENS=2
ADD_CODE_TOKENS=1
MINIMUM_FREQUENCY=0
BYTE_FALLBACK=1
CHARACTER_COVERAGE=0.9999
VOCAB_SIZE=128000
TRAIN_EXTREMELY_LARGE_CORPUS=1
ALPHA=-1


##################################################################################################
### TEST DATA (WITH CODE) ########################################################################
##################################################################################################

# anforanden
if [ 1 -eq 1 ]
then
  ListDatasetFiles=("data/books_sv_epub_100.jsonl")
  # ListDatasetFiles=("data/books_sv_epub_100_max4000.jsonl")
  # ListDatasetFiles=("data/books_sv_epub_100_max40000.jsonl")
  # ListDatasetFiles=("data/books_sv_epub_100_max400000.jsonl")
  ListDatasetName=("2")
  # LIBRARY="HF"
  # LIBRARY="SP"
  # ADD_PREFIX_SPACE=1
  # ADD_WHITESPACE_TOKENS=0
  # BYTE_FALLBACK=0
  # CHARACTER_COVERAGE=1.0
  VOCAB_SIZE=10000
fi

# anforanden + code
if [ 0 -eq 1 ]
then
  # ListDatasetFiles=("data/anforanden.jsonl data/code.json data/fibrec.json")
  ListDatasetFiles=("data/books_sv_epub_100.jsonl
                     data/code.json data/code.json data/code.json data/code.json data/code.json data/code.json
                     data/fibrec.json data/fibrec.json data/fibrec.json data/fibrec.json data/fibrec.json data/fibrec.json")
  ListDatasetName=("2p")
  VOCAB_SIZE=128000
  # ADD_PREFIX_SPACE=0
  # ADD_WHITESPACE_TOKENS=0
  # ADD_CODE_TOKENS=1
fi

##################################################################################################
### WIKI: SINGLE LANGUAGE ########################################################################
##################################################################################################
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("data/wiki_da_h1p.jsonl"
                    "data/wiki_en_h1p.jsonl"
                    "data/wiki_is_h1p.jsonl"
                    "data/wiki_no_h1p.jsonl"
                    "data/wiki_sv_h1p.jsonl")
  ListDatasetName=("3da" "3en" "3is" "3no" "3sv")
  VOCAB_SIZE=128000

  # ListDatasetFiles=("data/wiki_is_1p.jsonl")
  # ListDatasetName=("3is")
  # VOCAB_SIZE=100000
fi

##################################################################################################
### WIKI: ALL ##########################################################################################
##################################################################################################

# ALPHA=1.0
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("data/wiki_da_h1p.jsonl
                     data/wiki_en_h1p.jsonl
                     data/wiki_is_h1p.jsonl
                     data/wiki_no_h1p.jsonl
                     data/wiki_sv_h1p.jsonl")
  ListDatasetName=("3all")
  BYTE_FALLBACK=0
  CHARACTER_COVERAGE=1.0  # 0.9999
  VOCAB_SIZE=51200
  # VOCAB_SIZE=64000
  # VOCAB_SIZE=80000
  # VOCAB_SIZE=96000
  # VOCAB_SIZE=112000
  # VOCAB_SIZE=128000
  ALPHA=1.0
fi

# ALPHA=0.8
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("data/wiki_da.jsonl data/wiki_da.jsonl
                     data/wiki_en.jsonl
                     data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl
                     data/wiki_no.jsonl data/wiki_no.jsonl
                     data/wiki_sv.jsonl data/wiki_sv.jsonl")
  ListDatasetName=("3all")
  ALPHA=0.8
fi

# ALPHA=0.6
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("data/wiki_da.jsonl data/wiki_da.jsonl data/wiki_da.jsonl data/wiki_da.jsonl
                     data/wiki_en.jsonl
                     data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl data/wiki_is.jsonl
                     data/wiki_no.jsonl data/wiki_no.jsonl data/wiki_no.jsonl data/wiki_no.jsonl
                     data/wiki_sv.jsonl data/wiki_sv.jsonl data/wiki_sv.jsonl")
  ListDatasetName=("3all")
  ALPHA=0.6
fi

##################################################################################################
### FINAL100: SINGLE LANGUAGE ####################################################################
##################################################################################################
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("data/data_100_final/books/final_da.jsonl
                     data/data_100_final/conversational/final_da.jsonl
                     data/data_100_final/misc/final_da.jsonl
                     data/data_100_final/web_commoncrawl/final_da.jsonl
                     data/data_100_final/web_sources/final_da.jsonl")
  ListDatasetName=("4da")
fi

##################################################################################################
### RUNS #########################################################################################
##################################################################################################
if [ 1 -eq 1 ]
then
  for i in "${!ListDatasetFiles[@]}";
  do
      python script_train.py \
        --library $LIBRARY \
        --dataset_files ${ListDatasetFiles[i]} \
        --dataset_name ${ListDatasetName[i]} \
        --unicode_normalization $UNICODE_NORMALIZATION \
        --individual_digits $INDIVIDUAL_DIGITS \
        --add_prefix_space $ADD_PREFIX_SPACE \
        --add_whitespace_tokens $ADD_WHITESPACE_TOKENS \
        --add_code_tokens $ADD_CODE_TOKENS \
        --minimum_frequency $MINIMUM_FREQUENCY \
        --byte_fallback $BYTE_FALLBACK \
        --character_coverage $CHARACTER_COVERAGE \
        --train_extremely_large_corpus $TRAIN_EXTREMELY_LARGE_CORPUS \
        --vocab_size $VOCAB_SIZE \
        --alpha $ALPHA
  done
fi
