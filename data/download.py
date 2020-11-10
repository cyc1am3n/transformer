import sys
import os
import re
import tarfile
from xml.etree.ElementTree import parse

from requests import get
import pandas as pd

if __name__ == '__main__':
    FR = 'fr'
    EN = 'en'
    TRAIN_SET = 'train.tags.fr-en.'
    DEV_SET = 'IWSLT17.TED.dev2010.fr-en.'
    TEST_SET = [f'IWSLT17.TED.tst{year}.fr-en.' for year in range(2010, 2016)]

    URL = 'https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz'
    TGZ = 'fr-en.tgz'

    path = os.getcwd()
    if os.path.basename(path) != 'data':
        os.chdir('data')

    if os.path.isdir('./ORIG') is False:
        os.mkdir('./ORIG')
    if os.path.isdir('./PREP') is False:
        os.mkdir('./PREP')

    if os.path.isfile(f'./ORIG/{TGZ}'):
        print(f'{TGZ} already exists, skip download.')
    else:
        try:
            with open(f'./ORIG/fr-en.tgz', "wb") as file:
                response = get(URL)               
                file.write(response.content)
                print(f'{URL} is successfully downloaded.')
        except Exception as e:
            print(e)
            sys.exit(f'{URL} is not successfully downloaded.')

    if os.path.isdir('./ORIG/fr-en'):
        print('fr-en already exists, skipping extraction')
    else:
        tar = tarfile.open('./ORIG/fr-en.tgz', "r:gz")
        tar.extractall('./ORIG')
        tar.close()
        print(f'{TGZ} is successfully extracted.')

    print("pre-processing train data..")
    train_p = re.compile(r"""
        \s*<doc[^>]+>\s*\n|
        \s*<url>.+\n|
        \s*<keywords>.+</keywords>\s*\n|
        \s*<speaker>.+\n|
        \s*<talkid>.+\n|
        \s*<title>.+\n|
        \s*<description>.+\n|
        <reviewer.+\n|
        <translator.+\n|
        </doc>.*\n*|
        ^\s+|\s+$
        """, re.VERBOSE | re.MULTILINE)

    for x in [FR, EN]:
        with open(f'./ORIG/fr-en/{TRAIN_SET}{x}', 'r', encoding='UTF8') as f:
            b = f.read()
        with open(f'./PREP/train.{FR}-{EN}.{x}', 'w', encoding='UTF8') as f:
            f.write(re.sub(train_p, '', b))

    print('pre-processing valid data..')
    for x in [FR, EN]:
        text = []
        with open(f'./PREP/valid.{FR}-{EN}.{x}', 'w', encoding='UTF8') as f:        
            tree = parse(f'./ORIG/fr-en/{DEV_SET}{x}.xml')
            root = tree.getroot()
            for node in root.iter():
                segs = node.findall('seg')
                if segs:
                    text += segs
            f.write('\n'.join([seg.text.strip() for seg in text]))

    print('pre-processing test data..')
    for x in [FR, EN]:
        text = []
        with open(f'./PREP/test.{FR}-{EN}.{x}', 'w', encoding='UTF8') as f:
            for test_file in TEST_SET:
                tree = parse(f'./ORIG/fr-en/{test_file}{x}.xml')
                root = tree.getroot()
                for node in root.iter():
                    segs = node.findall('seg')
                    if segs:
                        text += segs
            f.write('\n'.join([seg.text.strip() for seg in text]))

    print('make CSV files for TabularDataset..')
    for dataset in ['train', 'valid', 'test']:
        with open(f'./PREP/{dataset}.fr-en.fr', 'r', encoding='UTF8') as fr, open(f'./PREP/{dataset}.fr-en.en', 'r', encoding='UTF8') as en:
            fr_lines = fr.readlines()
            en_lines = en.readlines()
            raw_data = {'en': [line for line in en_lines], 'fr' : [line for line in fr_lines]}
            df = pd.DataFrame(raw_data, columns=['en', 'fr'])
            df.to_csv(f'./PREP/{dataset}.csv', index=False)

    print('done!')
