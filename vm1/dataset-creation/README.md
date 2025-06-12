After getting your results from Med-R1 (base model was Qwen-VL-2B) and filtering it to be only the wrong answers,
we can perform the following steps.
These scripts assume that you have copied the following repositories to your $HOME.
If you have not, run `bash copy_repositories.sh` to clone the repositories locally.

- `create_batch_file.py`: This will create a single large batch file that contains the JSON file that you point to, formatted requests for the API you are pointing at.
- `split_batch_file.py`: If your batch file is too big, the dataset will be rejected by the Batch API. Instead, use this script to split and send multiple smaller batch files.
- `batch_reasoning_trace_gen.py`: This sends your batch file(s) to the Batch API. This will incur costs!
- `pull_and_review_batch_results.py`: This pulls the batches generated and generates a final dataset, use this after ALL batches are done.
