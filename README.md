### Main interesting files
- [Model definition](https://github.com/unic0rn9k/tokentango/blob/master/src/tokentango/model.py)
- [Training loop](https://github.com/unic0rn9k/tokentango/blob/master/src/tokentango/train.py)
- [Marimo notebook / script for running training](https://github.com/unic0rn9k/tokentango/blob/master/scripts/fake_news_bert.py)

### Sources
- [FakeNewsCorpus (training data)](https://github.com/several27/FakeNewsCorpus)
- ["For Perception Tasks: The Cost of LLM Pretraining by Next-Token Prediction Outweigh its Benefits" by Balestriero et al.](https://openreview.net/forum?id=wYGBWOjq1Q)
- [Reference for the pretrained model (also by me)](https://github.com/unic0rn9k/fake_news/blob/final_friday/bert_pretrained.ipynb)

### Comparison between pretrained and from-scratch
**Differences in setup:**
Training conditions where not the exact same.
- Pretrained model is larger, than the one I trained from scratch. The original bert models had model dimensions where tested for from-scratch training, but were found to be compute heavy, and generalize worse on contrived synthetic test case, along with FakeNewsCorpus.
- Pre-trained model was trained on raw data set, which had a bias in the labels / class imbalance, which may have negatively affected training, while also giving the impression of better performance on the test set.

**Results:**
|  | fine-tuning metric | from-scratch metric |
|-|-|-|
| duration     | 8 hours | 2 hours |
| bias corrected test accuracy | 90% | 93% |
| biased accuracy | 94% | N/A |

So fine-tuning again, with a evenly sampled training set would be a good idea. Also perhaps training from-scratch on a biased dataset, either method is more robus against sample bias.

### Other experiments
- CLS+MLM objective was tested against CLS only, which yielded no benifit in the MLM objective on the classification task.
- Using synthetic dataset, to probe for optimal model hyper-parameters and dimensions.

### Notes
Repo might not be entirely up to date. Newer changes can be found on the development branch.

### Preliminary results (newer graphs might be added later)
<img width="1136" height="540" alt="newplot(94)" src="https://github.com/user-attachments/assets/13ad0754-8786-48c1-86a0-ebd2e1a7bca3" />
<img width="1136" height="540" alt="newplot(93)" src="https://github.com/user-attachments/assets/29147569-5154-4cf4-b67c-7df0338e2369" />


