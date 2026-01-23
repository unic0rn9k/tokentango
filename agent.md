# General instructions
- Git commit and push after every successful task completion
- Don't use `git add .`, instead add all the files you have edited individually
- CRITICAL: Do NOT push changes until you have properly verified your work works correctly

# Notes
- Code execution duration: The data loading in `fake_news_bert.py` can take significant time because it loads from `995,000_rows.csv`
- To reduce data loading time, modify the `frac` parameter in `load_data(0.8)` call in `fake_news_bert.py`:
  - `0.1` loads ~100,000 rows
  - `0.05` loads ~50,000 rows
  - `0.01` loads ~10,000 rows

# TODO
- [x] add logic to `fake_news_bert.py` to make it load `checkpoint.pth`
    - if the checkpoint exists, then skip the training and go straight to model validation.

- [x] test and debug `fake_news_bert.py`
    - first, since the file is a marimo notebook, that has been directly converted from ipynb format, add some print statements, so you can tell whats going on.
    - use `uv run fake_news_bert.py` to run the notebook in console only mode. You should now see the output from your print statements.

- [ ] clean project root, there is a lot of unorganized files, find a better project structure
    1. use `find` to list current directory structure and consider how new structure should look
    2. create folders and move files
    3. update paths in any python files or other scripts (refrain from editing ipynb files)
    4. specifically verify that `fake_news_bert.py` still works
