==============================
gpt-sw3-tokenizer end-to-end tests
==============================

Step 1
......

copy end-to-end test environment file to base directory
::

    cp e2e_tests/env.ini .

Step 2
......
make sure there is a folder ``./e2e_data_original`` with the following files:
::

    ls e2e_data_original
    ## articles_en.jsonl
    ## books_hq_en.jsonl
    ## conversational_en.jsonl
    ## math_en.jsonl
    ## misc_en.jsonl
    ## web_commoncrawl_en.jsonl
    ## web_sources_en.jsonl

Step 3
......

run end-to-end test

::

    pytest e2e_tests/e2e_test.py

output files (stdout & stderr) can be found at ``./e2e_tests/e2e_test_data``

Step 4
......

run notebook ``e2e_test_sampling.ipynb``

::

    jupyter notebook
