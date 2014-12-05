import os
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy
import json
import operator
from gensim import corpora, models, similarities

from TextsDAO import TextsDAO
from CorpusDAO import DictionaryDAO
from CorpusDAO import CorpusDAO
from html.parser import HTMLParser

BASE_DIR = "data"
BASE_META_DIR = "data"
DB = os.path.join(BASE_META_DIR, "stackoverflow-posts.db")
SERIALIZED_CORPUS = os.path.join(BASE_META_DIR, "corpus.mm")
SERIALIZED_TFIDF = os.path.join(BASE_META_DIR, "tfidf.model")
SERIALIZED_TFIDF_CORPUS = os.path.join(BASE_META_DIR, "corpus_tfidf.mm")

SERIALIZED_LSI_CORPUS = os.path.join(BASE_META_DIR, "corpus_lsi.mm")
TOPICS_LIST = os.path.join(BASE_META_DIR, "list_topics.list")
SIMILARITY_INDEX = os.path.join(BASE_META_DIR, "similarity.index")
DOC_TO_TAG =  os.path.join(BASE_META_DIR, "doc_to_tag.npy")
TAG_MAP = os.path.join(BASE_META_DIR,"tag_to_id.dict")
TID_TO_TAG = os.path.join(BASE_META_DIR,"tid_to_tag.npy")

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def tokenize(string):
        """logic to tokenize a string. Involves removing tags, splitting, lowercasing words etc"""
        s = MLStripper()
        s.feed(string)
        new_s = ' '.join([x for x in re.sub('[^a-zA-Z0-9\n\.]', ' ', s.get_data()).split() if len(x) > 1])
        return ' '.join(new_s.split("\n")).split()

def main():
    dictionary = DictionaryDAO(BASE_META_DIR, BASE_DIR, DB).getDictionary()
    
    if os.path.isfile(SERIALIZED_CORPUS):
        corpus = corpora.MmCorpus(SERIALIZED_CORPUS)
    else:
        corpus_dao = CorpusDAO(BASE_META_DIR, BASE_DIR,DB)
        corpora.MmCorpus.serialize(SERIALIZED_CORPUS, corpus_dao)
        corpus = corpora.MmCorpus(SERIALIZED_CORPUS)

    # Confirm if its populated
    print(len(corpus))

    if os.path.isfile(SERIALIZED_TFIDF):
        tfidf = models.TfidfModel.load(SERIALIZED_TFIDF)
    else:
        tfidf = models.TfidfModel(corpus)
        tfidf.save(SERIALIZED_TFIDF)

    if os.path.isfile(SERIALIZED_TFIDF_CORPUS):
        corpus_tfidf = corpora.MmCorpus(SERIALIZED_TFIDF_CORPUS)
    else:
        corpus_tfidf = tfidf[corpus]
        corpora.MmCorpus.serialize(SERIALIZED_TFIDF_CORPUS, corpus_tfidf)

    print("loaded tfidf corpus")
    print(len(corpus_tfidf))
    
    if os.path.isfile(SERIALIZED_LSI_CORPUS):
        lsi = models.LsiModel.load(SERIALIZED_LSI_CORPUS)        
    else:
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
        lsi.save(SERIALIZED_LSI_CORPUS)

    """with open(TOPICS_LIST, "w") as f:
        print(lsi.show_topics(num_topics=-1, num_words=10, log=False, formatted=True), file=f)"""

    corpus_lsi = lsi[corpus_tfidf]
    print(len(corpus_lsi))


    if os.path.exists(SIMILARITY_INDEX):
        index = similarities.MatrixSimilarity.load(SIMILARITY_INDEX)
    else:
        index = similarities.MatrixSimilarity(corpus_lsi, num_features = 200)
        index.save(SIMILARITY_INDEX)

    # At this point we have created the similarity matrix 
    # we will start finding similarities between a new document
    # and existing documents, where the dimensions are topics.
    # similarity will be determined using the cosine distance between 
    # topic_vector for new document and the topic_vectors for existing documents
    # in the corpus.

    new_doc = tokenize("""
        When setting a form's opacity should I use a decimal or double?
        <p>I want to use a track-bar to change a form's opacity.</p>

        <p>This is my code:</p>

        <pre><code>decimal trans = trackBar1.Value / 5000;
        this.Opacity = trans;
        </code></pre>

        <p>When I try to build it, I get this error:</p>

        <blockquote>
          <p>Cannot implicitly convert type 'decimal' to 'double'.</p>
        </blockquote>

        <p>I tried making <code>trans</code> a <code>double</code>, but then the control doesn't work. This code has worked fine for me in VB.NET in the past. </p>
""")
    vec_bow = dictionary.doc2bow(new_doc)
    vec_lsi = lsi[vec_bow]
    #print("Doc vector in LSI space" + str(vec_lsi))

    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print("**********************")
    relevant_similar_documents = sims[0:10]
    #print(sims[0:10])

    # we have the similarity coefficient with us and a list of documents that 
    # have the highest similarity to our document.

    # lets load the dictionary and the document_to_tag matrix from the memory
    PID_TO_TAGS_LIST = numpy.load(DOC_TO_TAG).tolist()
    TID_TO_TAG_LIST = numpy.load(TID_TO_TAG).tolist()
    with open(TAG_MAP, "r") as f:
        TAG_TO_ID_DICT = json.loads(f.read())
    print("{}".format(len(PID_TO_TAGS_LIST)))
    print("{}".format(len(TAG_TO_ID_DICT)))

    TAG_ID_TO_COUNT = {}
    for doc_id, similartiy_coefficient in relevant_similar_documents:
        for tag_id in PID_TO_TAGS_LIST[doc_id]:
            TAG_ID_TO_COUNT[tag_id] = TAG_ID_TO_COUNT.get(tag_id, 0) + similartiy_coefficient
    sorted_tag_id_to_count = sorted(TAG_ID_TO_COUNT.items(), key=operator.itemgetter(1))
    highest_coeff = sorted_tag_id_to_count[-1][1]

    print("Total recommended tags" + str(len(sorted_tag_id_to_count)))
    #print(TID_TO_TAG_LIST)
    for tid, coeff in reversed(sorted_tag_id_to_count):
        print("Confidence: {:4.3f} Tag:{:10}".format((coeff/5), TID_TO_TAG_LIST[tid]))


    

if __name__ == "__main__":
    main()
