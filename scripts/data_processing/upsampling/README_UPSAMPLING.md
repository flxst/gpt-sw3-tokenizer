## Upsampling [TODO]

- ```
  python scripts/upsampling/script_upsampling.py
  --dataset_files <dataset_files>
  --stats <stats>
  --total <total>
  --alpha <alpha>  # upsampling parameter, 0 <= alpha <= 1
  ```
    - computes the upsampling factors for each dataset, using alpha parameter
    - write upsampling factors to `data/file-upsampled.json`

    - in addition to single runs,
      the bash script `bash upsampling.sh` allows to systematically
      execute multiple runs for the purpose of experimentation.

