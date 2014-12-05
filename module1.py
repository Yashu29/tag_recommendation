import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

from gensim import corpora, models, similarities

from TextsDAO import TextsDAO
from CorpusDAO import DictionaryDAO
from CorpusDAO import CorpusDAO



BASE_DIR = "."
BASE_META_DIR = "."
SERIALIZED_CORPUS = os.path.join(BASE_META_DIR, "corpus.mm")
SERIALIZED_TFIDF_CORPUS = os.path.join(BASE_META_DIR, "corpus_tfidf.mm")

def main():

    dictionary = DictionaryDAO(BASE_META_DIR, BASE_DIR).getDictionary()

    if os.path.isfile(SERIALIZED_CORPUS):
        corpus = corpora.MmCorpus(SERIALIZED_CORPUS)
    else:
        corpus = CorpusDAO(BASE_META_DIR, BASE_DIR)
        corpora.MmCorpus.serialize(SERIALIZED_CORPUS, corpus)

    #for key, value in corpus.getDictionary().items():
    #    print("Key:{} Value:{}".format(key, value))
    # Confirm if its populated
    #for vector in corpus:
    #    print(vector)
    if os.path.isfile(SERIALIZED_TFIDF_CORPUS):
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = corpora.MmCorpus(SERIALIZED_TFIDF_CORPUS)
    else:
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        corpora.MmCorpus.serialize(SERIALIZED_TFIDF_CORPUS, corpus_tfidf)
    
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
    corpus_lsi = lsi[corpus_tfidf]


    count = 0

    for doc in corpus_lsi:
        count += 1

    print("Length of corpus is " + str(count))

    document = """b'<c#><winforms><opacity><datediff>'
b"When setting a form's opacity should I use a decimal or double?"
b"<p>I want to use a track-bar to change a form's opacity.</p>\n\n<p>This is my code:</p>\n\n<pre><code>decimal trans = trackBar1.Value / 5000;\nthis.Opacity = trans;\n</code></pre>\n\n<p>When I try to build it, I get this error:</p>\n\n<blockquote>\n Given a specific <code>DateTime</code> value, how do I display relative time, like:</p>\n\n<ul>\n<li>2 hours ago</li>\n<li>3 days ago</li>\n<li>a month ago</li>\n</ul>\n\n<p>Et cetera?"
"""
    document = ''.join(e for e in document if e.isalnum() or e == ' ')
    new_doc_bow = dictionary.doc2bow(document.lower().split())
    #print(tfidf[new_doc_bow])

    new_corpus_tfidf = [tfidf[new_doc_bow]]
    lsi.add_documents(new_corpus_tfidf)
    new_corpus_lsi = lsi[new_corpus_tfidf]
    
    doc_count = 0
    topic_count = 0
    for new_doc in new_corpus_lsi:
        doc_count += 1
        print(new_doc)
        for topic_id, coorelation in new_doc:
            if coorelation > 0.05:
                lsi.print_topic(topic_id)
                topic_count += 1
    print("All docs: {} AND Related topics: {}".format(doc_count, topic_count))
    

if __name__ == "__main__":
    main()