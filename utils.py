'''
Utils for Textmining

Deze module bestaat uit de verschillende methodes die zijn ontwikkeld in het project textmining-digitalisering. De methodes zijn gescheiden op onderwerp en zijn vrij inzetbaar.


TODO
- Gebruik functie map() in de pipeline en map_to_func() method
'''


import string
import spacy
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

##################
# PDF EXTRACTION #
##################

def pdf_extractor(file):
    '''
    Deze functie neemt een file in .pdf format en converteert die naar .txt format
    :param file: pad string naar een .pdf file, String
    :return: lijst met string(s) die de text van de pdf bevat, List()
    '''
    pages_dict = {}
    pages_list = []
    
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.pdfdevice import PDFDevice
    from pdfminer.pdfpage import PDFTextExtractionNotAllowed
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTTextBoxHorizontal
    from pdfminer.converter import PDFPageAggregator
    
    try:
        base_path = ""
        password = ""
        
        fp = open(file, "rb")
        parser = PDFParser(fp)
        document = PDFDocument(parser, password)
        
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed

        rsrcmgr = PDFResourceManager()
        extr_params = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=extr_params)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        print('Extracten: {}'.format(file))
        
        for page_index, page in enumerate(PDFPage.create_pages(document)):
            extracted_text = ""
            interpreter.process_page(page)
            layout = device.get_result()
            
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    extracted_text += lt_obj.get_text()
                    extracted_text = extracted_text.encode('utf-8', errors='replace')
                    extracted_text = unicodedata.normalize('NFKD', extracted_text.decode('utf-8'))
                    
            pages_list.append(extracted_text)
            
        fp.close()
        
        return pages_list
    except:
        print('Niet mogelijk om {} te extracten'.format(file))
        return pages_list

##################
# TEXT SCRUBBING #
##################

def scrub(text):
    '''
    Deze functie neemt een string en schoont die op aan de hand van bepaalde regels
    :param text: String
    :return: String
    '''
    text = text.replace('"', '')
    text = text.replace('\\n', ' ')
    text = text.replace('\\t', ' ')
    unicode_index = [m.start() for m in re.finditer(r'\\u', text)]
    unicode_instances = [text[i:i+6].replace(' ', '') for i in unicode_index]
    for u in unicode_instances:
        text = str.replace(text, u, '')

    while text.count('  ') > 0:
        text = str.replace(text, '  ', ' ')

    return text


#################
# TEXT CLEANING #
#################

def lower(data):
    '''
    Take string, return string with only lower case

    :param data: string
    :return: string
    '''
    data = data.split()
    return ' '.join([d.lower() for d in data])

def remove_net(data):
    '''
    This function removes substrings containing .nl .net .com and @ from string
    :param data: string
    :return: string
    '''
    data = data.split()
    return ' '.join([d for d in data if not any(net in d for net in ['.nl', '.net', '.com', '@'])])

def remove_punctuation(data):
    '''
    This function removes punctuation from string
    :param data: string
    :return: string
    '''
    data = data.split()
    return ' '.join([d.translate(str.maketrans('', '', string.punctuation)) for d in data])

def remove_digit(data):
    '''
    This function removes digits from string
    :param data: string
    :return: string
    '''
    data = data.split()
    return ' '.join([d for d in data if not d.isdigit()])

def remove_dates(data):
    '''
    This function removes months and days from string
    :param data: string
    :return: string
    '''
    dates = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus', 'september', 'oktober', 'november', 'december', 
                'maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
    data = data.split()
    return ' '.join([d for d in data if not any(dt in d.lower() for dt in dates)])

def remove_stops(data):
    '''
    This function removes stopwords from string, stopwords defined by nltk
    :param data: string
    :return: string
    '''
    from nltk.corpus import stopwords
    stop_words = stopwords.words('dutch')
    data = data.split()
    return ' '.join([d for d in data if d not in stop_words])


##################################################
# MAP MULTIPLE FUNCTIONS TO DATA LIKE A PIPELINE #
##################################################

def map_to_func(data, funcs):
    '''
    This function applies a list of functions to a (nested) list of strings in series, order of functions in funcs
    :param data: nested list of strings, List
    :param funcs: list of functions, List
    :return: nested list of strings (same structure as input), List

    TODO: - Return intermediate results (Make generator/use yield?)
    '''
    if type(data) is list: return [map_to_func(d, funcs) for d in data]
    else:
        if type(funcs) is not list: funcs = [funcs]
        for f in funcs:
            data = f(data)
        return data


########################
# PREPROCESS TEXT DATA #
########################

def tokenize(data):
    '''
    This function tokenizes the string
    :param data: string
    :return: string
    '''
    return data.split() 

def lemmatize(text, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    '''
    This function lemattizes a string
    :param text: String
    :param allowed_postages: list of allowed postages (strings) given nlp package, List()
    :return: list of strings, List()
    '''
    nlp = spacy.load("nl_core_news_lg", disable=["parser", "ner"])
    doc = nlp(text)
    return [d.lemma_ for d in doc if d.pos_ in allowed_postags]

def ngram(tokens, n):
    '''
    Take list of tokens, return list of ngrams in tuples
    :param tokens: list of strings, List()
    :param n: integer for n of ngram, Int
    :return: list of tuples with strings, List()
    '''
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def ngrams(tokens, n):
    '''
    Take list of lists of tokens, return list of lists of ngrams in tuples
    :param tokens: list, or list of lists etc. of strings, List()
    :param n: integer for n of ngram, Int
    :return: list of tuples with strings, List()
    '''
    if type(tokens[0]) is list: return [ngrams(t, n) for t in tokens]
    else: return ngram(tokens, n)

def onehot_matrix(tokens):
    '''
     This function takes a list of tokens and returns a one-hot-encoded sparse matrix
     :param tokens: list of tokens/strings, List()
     :return: Pandas DataFrame
    '''
    vocab = sorted(set(tokens))
    num_tokens = len(tokens)
    vocab_size = len(vocab)
    onehot_vectors = np.zeros((num_tokens, vocab_size), int)
    for i, word in enumerate(tokens):
        onehot_vectors[i, vocab.index(word)] = 1
    ' '.join(vocab)
    return pd.DataFrame(onehot_vectors, columns=vocab)

def tf_idf(tokens, ngram_range=(1, 1)):
    '''
    Take list of tokens return pandas dataframe of tokens with tf-idf score
    :param tokens: list of tokens, List()
    :param ngram_range: tuple of integers for range of ngrams to return, Tuple()
    :return: pandas.DataFrame, dict()
    '''
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(tokens)
    df = pd.DataFrame(tfidf_matrix.T.todense(), index=vectorizer.get_feature_names())
    return df, vectorizer.vocabulary_