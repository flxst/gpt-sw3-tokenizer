# --- needs to be the same as in DATA_TRAIN.sh ---
NAME_ALL = "4all-a1.0"

# ---
DATA_EVAL = [
    "wiki_da_t1p.jsonl",
    "wiki_en_t1p.jsonl",
    "wiki_is_t1p.jsonl",
    "wiki_no_t1p.jsonl",
    "wiki_sv_t1p.jsonl",
]

# --- last entry needs to be the same as in DATA_TRAIN.sh ---
# VOCAB_SIZES = [51200, 64000, 80000, 96000, 112000, 128000]
VOCAB_SIZES = [10000, 20000]
