DATASET_FILES="wiki_da.jsonl wiki_en.jsonl wiki_is.jsonl wiki_no.jsonl wiki_sv.jsonl"
STATS="wiki_stats.json"
TOTAL="wiki_total.jsonl"

### OPTIONS ####
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.9
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.8
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.7
python script_upsampling.py --dataset_files $DATASET_FILES --stats $STATS --total $TOTAL --alpha 0.6
