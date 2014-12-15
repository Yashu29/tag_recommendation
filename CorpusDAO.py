from gensim import corpora, models, similarities

import os
from TextsDAO import TextsDAO
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from nltk.corpus import stopwords
STOP_WORDS = stopwords.words("english")


class CorpusDAO(object):
    """Memory friendly corpus"""
    def __init__(self, base_metadata_dir, base_data_dir, db):
        logging.info("Loading corpus")
        self.__metadata_dir = base_metadata_dir
        self.__data_dir = base_data_dir
        self.__db = db
        self.__dictionary = DictionaryDAO(base_metadata_dir, base_data_dir,db).getDictionary()

    def __iter__(self):
        iterator_text = TextsDAO(self.__data_dir, self.__db)
        for post in iterator_text:
            print("C..")
            yield self.__dictionary.doc2bow(post)
        iterator_text.close()

    def getDictionary(self):
        return self.__dictionary

class DictionaryDAO(object):
    """abstracts out the details of the dictionary. if already processed, the dictionary is loaded from memory"""
    def __init__(self, base_metadata_dir, base_dir, db):
        self.__base_dir = base_dir
        self.__db = db
        if os.path.isfile(os.path.join(base_metadata_dir, "dictionary.dict")):
            self.__dictionary = corpora.Dictionary.load(os.path.join(base_metadata_dir, "dictionary.dict"))
        else:
            self.__dictionary = self.createDictionary()
            self.__dictionary.save(os.path.join(base_metadata_dir, "dictionary.dict"))

    def createDictionary(self):
         iterator_text = TextsDAO(self.__base_dir, self.__db)
         dictionay = corpora.Dictionary(line for line in iterator_text)
         low_freq_words = [tokenid for tokenid,docfreq in dictionay.dfs.items() if docfreq == 1]
         dictionay.filter_tokens(low_freq_words + STOP_WORDS)
         dictionay.compactify()
         return dictionay


    def getDictionary(self):
        return self.__dictionary


class TestCorpusDAO(object):
    def __init__(self, base_metadata_dir, base_dir, db):
        self.__base_dir = base_dir
        self.__data_dir = base_dir
        self.__db = db
        self.__dictionary = DictionaryDAO(base_metadata_dir, base_dir, db).getDictionary()

    def __iter__(self):
        iterator_text = TextsDAO(self.__data_dir, self.__db, get_both=True)
        count = 0
        for post,tags in iterator_text:
            count+=1
            if count%10 == 0:
                yield (self.__dictionary.doc2bow(post),tags)
        iterator_text.close()

