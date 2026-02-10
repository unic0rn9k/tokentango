print("Token", end="")

from .model import BertClassifier
from .data import BertData
from .config import TrainingConfig, Checkpoint, EvaluationResult
from . import fake_news
from . import train

print("Tango")
