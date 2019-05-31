from datetime import datetime
from elasticsearch import Elasticsearch
import codecs
import config

corpus_path = config.corpus_path
index_name = 'corpus'
doc_type = 'openbook'
id_count, corpus_total = 0, 1326

if __name__ == "__main__":
    es = Elasticsearch()
    es.indices.create(index=index_name)

    source = []
    with codecs.open(corpus_path, 'r', 'utf-8') as corpus_f:
        for line in corpus_f:
            id_count += 1
            if id_count % 100 == 0:
                print("Processing: {} / {}. Ratio: {}".format(id_count, corpus_total, round(id_count * 1.00 / corpus_total, 2)))
            sentence = line.strip()
            source.append({"index": {}})
            source.append({"text": sentence})
    
    res = es.bulk(index=index_name, doc_type=doc_type, body=source)