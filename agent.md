# Project structure
```text
.gitignore
.python-version
README.md
accuracy_benchmark_results.txt
agent.md
checkpoint_test_results.csv
data/
data/995,000_rows.csv
data/bpe_tokenizer.json
data/checkpoints/
experiments/
experiments/brute_convergence/
notebooks/
notebooks/diffusion.ipynb
notebooks/fake_news_bert.ipynb
notebooks/trivial_bert.ipynb
notebooks/very_experimental.ipynb
playlist
pyproject.toml
scripts/
scripts/fake_news_bert.py
scripts/test_checkpoints.py
src/
src/tokentango/
src/tokentango/__init__.py
src/tokentango/bert_from_scratch.py
src/tokentango/fake_news.py
src/tokentango/main.py
src/tokentango/model.py
src/tokentango/train.py
uv.lock
```

# General instructions
- Git commit and push after every successful task completion
- Don't use `git add .`, instead add all the files you have edited individually
- Do NOT push changes until you have properly verified your work works correctly. That means, running a test script, like `fake_news_bert.py`, and verifying the output.

- Code execution duration: The data loading in `fake_news_bert.py` can take significant time because it loads from `995,000_rows.csv`
- To reduce data loading time, modify the `frac` parameter in `load_data(frac)` call in `fake_news_bert.py`, where frac can be 0.8 for a full test (almost. don't use 1 tho), and 0.01 for something that should just test code logic, and not model performance
- Set shell command execution timeout appropriately depending on usecase:
    - retraining a model can take multiple hours, depending on epochs etc,
    - validating a model can take 15-30 min,
    - and testing code logic with small fraction of training data should only take 5 min at most.
- Don't speak or think in Chinese
- Don't kill running processes. If you get an Nvidia "out of memory" error, sleep in a loop til the process exits
- Don't edit any part in agent.md, that you weren't explicitly told to. Fx don't update the formatting of unrelated todo items etc.

- Training and validation fractions:
    - Use `frac=0.01` when testing code logic (should complete in ~5 min)
    - Use `frac=0.75` for full training runs
    - Always set `random_state` for reproducibility during experiments
    - Run experiments with different seeds (e.g., 42, 69, 123) to verify results are consistent


# Checkpoint Selection Feature
The checkpoint selection feature allows you to control model checkpoint loading through the `MODEL_CHECKPOINT_PATH` environment variable when running in headless mode (console execution).

## train mode
Starts training from scratch without loading any checkpoint.
```bash
MODEL_CHECKPOINT_PATH=train uv run scripts/fake_news_bert.py
```

## latest mode (default)
Automatically finds and loads the most recently modified checkpoint from `data/checkpoints/` directory. If no checkpoint exists, training starts from scratch.
```bash
MODEL_CHECKPOINT_PATH=latest uv run scripts/fake_news_bert.py
# or simply:
uv run scripts/fake_news_bert.py
```

## specific path mode
Loads a checkpoint from a specific file path. If the file doesn't exist, training starts from scratch.
```bash
MODEL_CHECKPOINT_PATH=data/checkpoints/checkpoint_2026-01-23_20-54-31_50.00.pth uv run scripts/fake_news_bert.py
```


# Implementation Details
The feature is implemented with two helper functions in `src/tokentango/train.py`:
- `list_checkpoints(checkpoints_dir)`: Returns a sorted list of checkpoints (newest first)
- `load_checkpoint(model, checkpoint_path)`: Loads checkpoint and returns metadata including epoch, accuracy, and training fraction

Checkpoint metadata is displayed after loading, including:
- Epoch number
- Checkpoint accuracy
- Training fraction used during checkpoint creation

# TODO
- [x] add logic to `fake_news_bert.py` to make it load `checkpoint.pth`
    - if the checkpoint exists, then skip the training and go straight to model validation.

- [x] test and debug `fake_news_bert.py`
    - first, since the file is a marimo notebook, that has been directly converted from ipynb format, add some print statements, so you can tell whats going on.
    - use `uv run fake_news_bert.py` to run the notebook in console only mode. You should now see the output from your print statements.

- [x] better accuracy scoring
    - rewrite the `tokentango.train.test_accuracy` method to take the fraction of validation data to testa against as an argument
    - replace other instances of code that computes validation accuracy with a call to this method

- [x] Use batching in accuracy score
    - test execution time of current implementation, and save it to a file
    - implement batching in the method for potential performance speed-up
    - test execution time again, and append to file
    - if it is slower, revert your changes and mark todo item as done
    - if it is faster, keep your changes and mark todo item as done

 - [x] Add confusion matrix to `fake_news_bert.py`
    - implement confusion matrix or use one from library
    - print confusion matrix to console with labels for axis, columns and rows
    - verify output of confusion matrix against result from `tokentango.train.test_accuracy`

- [x] Update training method, to not override model checkpoints
    - Use the date, time and accuracy in the name of the checkpoint file
    - load the newest checkpoint file, if there are any

- [x] Update training progression (train.py)
    - add a cool progress bar (don't add new libraries for this)
    - move the checkpoint saving into same part that prints "ta: ...%"

- [x] Add information about training set fraction to checkpoint metadata
    - Make sure that existing checkpoints can still be loaded, even tho they don't have this field

 - [x] Add checkpoint selection - headless mode
     - add environment variable check for MODEL_CHECKPOINT_PATH
     - implement checkpoint loading logic (train/latest/specific path)
     - unwrap and update commented checkpoint loading code
     - add helper function to detect and list available checkpoints

- [x] Add checkpoint selection - web UI mode
    - search web for marimo dropdown widget documentation
    - add marimo dropdown widget with train/latest/specific options
    - integrate widget value with training flow control
    - make selection work consistently between UI and env var

- [x] Fix warnings when running `fake_news_bert.py`
    - note warnings when running training (use small frac)
    - search for documentation on specific warnings
    - fix warnings

- [x] Update Project structure section

- [x] Define BertData data class in `src/tokentango/data.py`

- [x] Update files to use BertData
    - Use BertData in `src/tokentango/train.py`
    - Use BertData in `src/tokentango/fake_news.py`
    - Use BertData in `scripts/fake_news_bert.py`
    - Use BertData in `scripts/test_checkpoints.py`

- [x] Load tokenizer if file already exists

- [x] Test bert-pretraining workflow
    - Check current date
    - Run `scripts/fake_news_bert.py` for training new checkpoints, stopping once training accuracy surpasses 80% for 3 iterations in a row.
    - Update checkpoints used in test_checkpoints.py to match new checkpoints (refer to current date)
    - Run `scripts/test_checkpoints.py`

- [ ] Implement TrainingConfig and Checkpoint dataclasses
    - create TrainingConfig dataclass with fields: train_frac, batch_size, lr, optimizer_type, use_mlm, seed, device, run_name (cute unique ID, inherited when resuming), checkpoint_dir
    - add from_env() classmethod to load from TT_ prefixed environment variables (TT_TRAIN_FRAC, TT_OPTIMIZER_TYPE, TT_USE_MLM, TT_RUN_NAME, etc.)
    - add validate() method to check config values
    - create Checkpoint dataclass with fields: model_state, optimizer_state, config (TrainingConfig), epoch, accuracy, timestamp, cls_losses, mlm_losses (losses accumulated since last checkpoint save)
    - create EvaluationResult dataclass with fields: accuracy, num_samples, confusion_matrix
    - update train(model, data, config: TrainingConfig) -> Checkpoint to save losses per iteration and return final checkpoint
    - update test_accuracy(...) -> EvaluationResult
    - update load_checkpoint() to return tuple[Checkpoint, model_state]
    - update list_checkpoints() to return list[Checkpoint]
    - ensure backward compatibility with old checkpoint format
    - update fake_news_bert.py to use TrainingConfig.from_env() when MODEL_CHECKPOINT_PATH=train

- [ ] Create checkpoint inspection script
    - create scripts/inspect_checkpoints.py
    - list all checkpoints with formatted display of config fields (run_name, optimizer_type, use_mlm, etc.) and metadata (epoch, accuracy, timestamp)
    - add command line flags: --sort accuracy|timestamp, --optimizer, --use-mlm, --min-accuracy
    - support aggregating loss histories across checkpoints for a given run_name

- [ ] Experiment: test_accuracy on source vs masked tokens
    - use existing checkpoints to avoid retraining
    - modify test_accuracy to accept token_type parameter ('source' or 'masked')
    - run test_accuracy with source_tokens on existing checkpoint
    - run test_accuracy with masked_tokens on same checkpoint
    - compare results

- [ ] Experiment: MLM objective ablation study
    - check existing checkpoints for train_frac used and presence of use_mlm in metadata
    - if insufficient checkpoints exist, train with use_mlm=True and save checkpoint with metadata
    - train with use_mlm=False and save checkpoint with metadata
    - compare test_accuracy between checkpoints

- [ ] Experiment: optimizer comparison (Adam, AdamW, SGD)
    - check existing checkpoints for optimizer_type in metadata
    - if insufficient checkpoints exist, train with Adam optimizer and save checkpoint
    - train with AdamW optimizer and save checkpoint
    - train with SGD optimizer and save checkpoint
    - compare test_accuracy and convergence speed across all three

- [ ] Replace progress bar with tqdm
    - uv add tqdm
    - replace progress bar with tqdm
    - add instructions to disable progress bar when run by agent

- [ ] Use float16 everywhere
    - remve amp code
    - ensure model is float16
    - ensure data is cast to float16 or loaded as float16

- [ ] Deterministic test accuracy
    - if the frac is 1, then the function should be deterministic
    - create file for unit test in scripts/test_test_accuracy.py
    - unit test should choose target list of accuracies first, then create model that always produces same output and test sets that will produce target accuracy if method is implemented correctly
    - test unit test unit test unit test :)

- [ ] Fix notebook UI
    - checkpoint_selector, checkpoint_path_input and mode_panel are defined in the marimo notebook, but not used anywhere
    - search for documentation on how to use marimo notebook widgets
    - fix notebook UI
