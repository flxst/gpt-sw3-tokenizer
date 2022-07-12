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
VOCAB_SIZE=64000
TRAIN_EXTREMELY_LARGE_CORPUS=1
ALPHA=-1


##################################################################################################
### 2. TEST DATA (WITH CODE) #####################################################################
##################################################################################################

# anforanden
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("books_sv_epub_100.jsonl")
  # ListDatasetFiles=("books_sv_epub_100_max4000.jsonl")
  # ListDatasetFiles=("books_sv_epub_100_max40000.jsonl")
  # ListDatasetFiles=("books_sv_epub_100_max400000.jsonl")
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
  # ListDatasetFiles=("anforanden.jsonl code.json fibrec.json")
  ListDatasetFiles=("books_sv_epub_100.jsonl
                     code.json code.json code.json code.json code.json code.json
                     fibrec.json fibrec.json fibrec.json fibrec.json fibrec.json fibrec.json")
  ListDatasetName=("2p")
  VOCAB_SIZE=128000
  # ADD_PREFIX_SPACE=0
  # ADD_WHITESPACE_TOKENS=0
  # ADD_CODE_TOKENS=1
fi

##################################################################################################
### 3. WIKI: SINGLE LANGUAGE #####################################################################
##################################################################################################
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("wiki_da_h1p.jsonl"
                    "wiki_en_h1p.jsonl"
                    "wiki_is_h1p.jsonl"
                    "wiki_no_h1p.jsonl"
                    "wiki_sv_h1p.jsonl")
  ListDatasetName=("3da" "3en" "3is" "3no" "3sv")
  VOCAB_SIZE=128000

  # ListDatasetFiles=("wiki_is_1p.jsonl")
  # ListDatasetName=("3is")
  # VOCAB_SIZE=100000
fi

##################################################################################################
### 3. WIKI: ALL #################################################################################
##################################################################################################
# ALPHA=1.0
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("wiki_da_h1p.jsonl
                     wiki_en_h1p.jsonl
                     wiki_is_h1p.jsonl
                     wiki_no_h1p.jsonl
                     wiki_sv_h1p.jsonl")
  ListDatasetName=("3all")
  # BYTE_FALLBACK=0
  # CHARACTER_COVERAGE=1.0  # 0.9999
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
  ListDatasetFiles=("wiki_da.jsonl wiki_da.jsonl
                     wiki_en.jsonl
                     wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl
                     wiki_no.jsonl wiki_no.jsonl
                     wiki_sv.jsonl wiki_sv.jsonl")
  ListDatasetName=("3all")
  ALPHA=0.8
fi

# ALPHA=0.6
if [ 0 -eq 1 ]
then
  ListDatasetFiles=("wiki_da.jsonl wiki_da.jsonl wiki_da.jsonl wiki_da.jsonl
                     wiki_en.jsonl
                     wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl wiki_is.jsonl
                     wiki_no.jsonl wiki_no.jsonl wiki_no.jsonl wiki_no.jsonl
                     wiki_sv.jsonl wiki_sv.jsonl wiki_sv.jsonl")
  ListDatasetName=("3all")
  ALPHA=0.6
fi

##################################################################################################
### 4. REAL DATASETS #############################################################################
##################################################################################################
if [ 1 -eq 1 ]
then
  # CHANGE THIS START !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  TOKENIZER_NUMBER=2

  DATASET_FILTER_DA="_da"
  DATASET_FILTER_IS="_is"
  DATASET_FILTER_EN="_en"
  DATASET_FILTER_NO="_no"
  DATASET_FILTER_SV="_sv"
  DATASET_FILTER_CD="_cd"
  DATASET_FILTER_ALL="all"

  # VOCAB_SIZE=64000
  # CHANGE THIS END !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  # single language name needs to be int + char + char!
  # multi  language name needs to be int + all-a?.?!
  NAME_DA=$TOKENIZER_NUMBER"da"
  NAME_IS=$TOKENIZER_NUMBER"is"
  NAME_EN=$TOKENIZER_NUMBER"en"
  NAME_NO=$TOKENIZER_NUMBER"no"
  NAME_SV=$TOKENIZER_NUMBER"sv"
  NAME_CD=$TOKENIZER_NUMBER"cd"
  NAME_ALL=$TOKENIZER_NUMBER"all-a1.0"

  echo "DATASET_FILTER_DA: " ${DATASET_FILTER_DA}
  echo "DATASET_FILTER_IS: " ${DATASET_FILTER_IS}
  echo "DATASET_FILTER_EN: " ${DATASET_FILTER_EN}
  echo "DATASET_FILTER_NO: " ${DATASET_FILTER_NO}
  echo "DATASET_FILTER_SV: " ${DATASET_FILTER_SV}
  echo "DATASET_FILTER_CD: " ${DATASET_FILTER_CD}
  echo "DATASET_FILTER_ALL:" ${DATASET_FILTER_ALL}
  echo ""
  echo "VOCAB_SIZE:  " ${VOCAB_SIZE}
  echo ""
  ListDatasetFiles=("all"
                    "all"
                    "all"
                    "all"
                    "all"
                    "all"
                    "all"
                    )
  ListDatasetFilters=("${DATASET_FILTER_DA}"
                      "${DATASET_FILTER_IS}"
                      "${DATASET_FILTER_EN}"
                      "${DATASET_FILTER_NO}"
                      "${DATASET_FILTER_SV}"
                      "${DATASET_FILTER_CD}"
                      "${DATASET_FILTER_ALL}"
                   )
  ListDatasetName=("${NAME_DA}"
                   "${NAME_IS}"
                   "${NAME_EN}"
                   "${NAME_NO}"
                   "${NAME_SV}"
                   "${NAME_CD}"
                   "${NAME_ALL}"
                   )
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
        --tokenizer_name ${ListDatasetName[i]} \
        --dataset_files ${ListDatasetFiles[i]} \
        --dataset_filter ${ListDatasetFilters[i]} \
        --unicode_normalization ${UNICODE_NORMALIZATION} \
        --individual_digits ${INDIVIDUAL_DIGITS} \
        --add_prefix_space ${ADD_PREFIX_SPACE} \
        --add_whitespace_tokens ${ADD_WHITESPACE_TOKENS} \
        --add_code_tokens ${ADD_CODE_TOKENS} \
        --minimum_frequency ${MINIMUM_FREQUENCY} \
        --byte_fallback ${BYTE_FALLBACK} \
        --character_coverage ${CHARACTER_COVERAGE} \
        --train_extremely_large_corpus ${TRAIN_EXTREMELY_LARGE_CORPUS} \
        --vocab_size ${VOCAB_SIZE} \
        --alpha ${ALPHA}
  done
fi
