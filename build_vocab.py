import json
import sentencepiece as spm

from utils import Tokenizer

if __name__ == '__main__':
    with open('./data/config.json', 'r') as f:
        config = json.load(f)
    print('train sentencepiece tokenizer..')
    for lang in ['en', 'fr']:
        tokenizer = Tokenizer(config['spm'][lang])
        tokenizer.fit_tokenizer()
    print()
    print('done!')
