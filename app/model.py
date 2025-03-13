
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import BartTokenizer, BartForConditionalGeneration

nlp = spacy.load('en_core_web_lg')




base = '/Users/adityavs14/Documents/Internship/Pianalytix/Month_2/sym2dis/app'


def summ1(text):
    doc = nlp(text)
    keywords = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['ADJ','PROPN','NOUN','VERB']
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keywords.append(token.text)
    freq_word = Counter(keywords)
    max_freq = Counter(keywords).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = freq_word[word]/max_freq

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    summarized = nlargest(3,sent_strength, key=sent_strength.get)
    return (' '.join([w.text for w in summarized]))


def summ2(text):
    model_name = 'facebook/bart-large-cnn'
    input_text = text

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.batch_encode_plus(
        [input_text], 
        return_tensors='pt', 
        max_length=1024, 
        truncation=True
    )
    summary_ids = model.generate(
        inputs['input_ids'], 
        num_beams=4, 
        max_length=100, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary

