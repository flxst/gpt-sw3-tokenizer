DATASET_FILES="data/wiki_da.jsonl data/wiki_en.jsonl data/wiki_is.jsonl data/wiki_no.jsonl data/wiki_sv.jsonl"
STATS="data/wiki_stats.json"
TOTAL="data/wiki_total.jsonl"

### OPTIONS ####
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.9
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.8
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.7
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.6
