import os
from typing import Dict, List
import sentencepiece as spm

class Tokenizer(object):
    """Tokenizer class"""
    def __init__(self, config: Dict[str, str]) -> None:
        """Instantiating Tokenizer class

        Args:
            config (Dict[str, str]): config for tokenizer which has attributes (input_data/model_prefix/vocab_size/tokenizer type).
        """
        self.data = config['input']
        self.model = config['model']
        self.vocab_size = config['vocab_size']
        self.type = config['type']

        self.sp = spm.SentencePieceProcessor()
        if os.path.isfile(f"{self.model}.model"):
            self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        """Load Sentencepiece tokenizer model"""
        model_dir = f"{self.model}.model"
        self.sp.load(model_dir)

    def fit_tokenizer(self) -> None:
        """Train and save Sentencepiece tokenizer model"""
        templates = "--input={} --model_prefix={} --vocab_size={} --model_type={}"
        cmd = templates.format(self.data, self.model, self.vocab_size, self.type)
        spm.SentencePieceTrainer.Train(cmd)

    def transform(self, sentence: str) -> List[str]:
        """Tokenize given sentence

        Args:
            sentence (str): sentence for tokenization
        """
        return self.sp.encode_as_pieces(sentence)
