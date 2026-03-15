### Sources and inspiration
- [FakeNewsCorpus (training data)](https://github.com/several27/FakeNewsCorpus) - training data used.
- ["For Perception Tasks: The Cost of LLM Pretraining by Next-Token Prediction Outweigh its Benefits" by Balestriero et al.](https://openreview.net/forum?id=wYGBWOjq1Q) - This article was the inspiration for this experiment, and explores a similar comparison of fine-tuning and from-scratch training, though it explores autoregresive decoder models instead of BERT, and also test across multiple large datasets and models.
- [BERT fine-tuned on FakeNewsCorpus by my uni study group and I](https://github.com/unic0rn9k/fake_news/blob/final_friday/bert_pretrained.ipynb) - reference fine-tuned model

### Comparison between pretrained and from-scratch
**Differences in setup:**
Training conditions where not the exact same.
- The model used for finetuning was distilbert-base-uncased.
- The pre-trained model is larger, than the one I trained from scratch. The dimensions used in the original BERT paper, where tested for from-scratch training, but were found to be compute heavy, and generalize worse on contrived synthetic test case, along with FakeNewsCorpus.
- Pre-trained model was trained on raw data set, which had a bias in the labels / class imbalance, which may have negatively affected training, while also giving the impression of better performance on the test set.
- The fine-tuning was done on a much smaller fraction of the dataset, than the from-scratch training. Since training from scratch with a smaller mode used so much less compute, it was viable to train on more data.

**Results:**
| Metric | Fine-Tuned | From-Scratch |
|-|-|-|
| Training samples used | 99.5k | 796k |
| Duration     | 8 hours | 2 hours |
| Bias corrected test accuracy | 90% | 92% |
| Biased accuracy | 94% | N/A |

So fine-tuning again, with a evenly sampled training set would be a good idea. Also perhaps training from-scratch on a biased dataset, either method is more robust against sample bias.
It would also have been interesting to explore how well the from-scratch model would have generalized, given a smaller amount of training data.

### Other experiments
- CLS+MLM objective was tested against CLS only, which yielded no benifit in the MLM objective on the classification task.

<img width="1616" height="540" alt="newplot(104)" src="https://github.com/user-attachments/assets/dc1371a4-a342-4bdd-ba13-c351e68ad7be" />

### Main interesting files, if you wan't too look at the code
**NOTE:** Repo might not be entirely up to date. Newer changes can be found on the other branches.
- [Model definition](https://github.com/unic0rn9k/tokentango/blob/master/src/tokentango/model.py)
- [Training loop](https://github.com/unic0rn9k/tokentango/blob/master/src/tokentango/train.py)
