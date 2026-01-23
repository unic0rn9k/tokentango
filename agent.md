# Project structure
```text
README.md
accuracy_benchmark_results.txt
agent.md
experiments/
notebooks/
notebooks/diffusion.ipynb
notebooks/fake_news_bert.ipynb
notebooks/trivial_bert.ipynb
notebooks/very_experimental.ipynb
playlist
pyproject.toml
scripts/
scripts/fake_news_bert.py
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
- Set shell command execution timeout appropriatly depending on usecase:
    - retraining a model can take multiple hours, depending on epochs etc,
    - validating a model can take 15-30 min,
    - and testing code logic with small fraction of training data should only take 5 min at most.
- Don't speak or think in chinese
- Don't kill running processes. If you get an nvidia "out of memory" error, sleep in a loop til the process exits

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

 - [ ] Add checkpoint selection - headless mode
    - add environment variable check for MODEL_CHECKPOINT_PATH
    - implement checkpoint loading logic (train/latest/specific path)
    - unwrap and update commented checkpoint loading code
    - add helper function to detect and list available checkpoints

 - [ ] Add checkpoint selection - web UI mode
    - search web for marimo dropdown widget documentation
    - add marimo dropdown widget with train/latest/specific options
    - integrate widget value with training flow control
    - make selection work consistently between UI and env var

 - [ ] Verify checkpoint with >80% accuracy against 0.8 fraction of training data

