import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

from tqdm import tqdm
import time

if __name__ == '__main__':


    sentences_df = pd.read_csv('../../data/sentences.csv')
    sentences_df.set_index('article_id', inplace=True)


    tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic-ner")
    model = AutoModelForTokenClassification.from_pretrained("classla/bcms-bertic-ner").to('cuda')

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device=1)

    sentence_ner_df = pd.DataFrame()

    for idx, row in tqdm(sentences_df.iterrows()):

        ner_result = ner_pipeline(row['sentence'])

        if ner_result:

            for ner in ner_result:

                sentence_ner_df['article_id'] = idx
                sentence_ner_df =  sentence_ner_df.append({'article_id':idx,
                                                            'sentence': row['sentence'],
                                                            'ner': ner['word'],
                                                            'ner_type': ner['entity_group'],
                                                            'ner_begin': int(ner['start']),
                                                            'ner_end': int(ner['end'])}, 
                                                            ignore_index=True)

    sentence_ner_df.to_csv('../../data/ner_sentences.csv', index=False)
        