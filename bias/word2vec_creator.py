import logging
import pickle

import gensim
from gensim.models import fasttext
import numpy as np
from tqdm import tqdm

from nltk import word_tokenize

import senticnetify
import util

model = None


# shaping:
#   - sequence (keep separated)
#   - avg (average the w2v)
# TODO: add turning into square or flattening
def run_w2v(df, output, shaping="sequence", word_limit=-1, sentics=False, model_only=False, zero_pad=False):
    global model

    if model is None:
        logging.info("Loading word2vec model...")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../data/raw/models/GoogleNews-vectors-negative300.bin", binary=True
        )

    return vectorize_collection(df, output, model, shaping, word_limit, sentics, model_only, zero_pad)


def run_glove(df, output, shaping="sequence", word_limit=-1, sentics=False, model_only=False, zero_pad=False):
    global model

    if model is None:
        logging.info("Loading glove model...")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../data/cache/models/glove2word2vec_pretrained.model"
        )

    return vectorize_collection(df, output, model, shaping, word_limit, sentic, model_only, zero_pad)


def run_fasttext(df, output, shaping="sequence", word_limit=-1, sentics=False, model_only=False, zero_pad=False):
    global model

    if model is None:
        logging.info("Loading fasttext model...")

        model = gensim.models.fasttext.load_facebook_vectors('../data/raw/models/wiki.en.bin')

    return vectorize_collection(df, output, model, shaping, word_limit, sentic, model_only, zero_pad)


def clear_model():
    global model
    del model
    model = None


# (model_only=False, zero_pad=False) = normal sentic usage
# (model_only=True, zero_pad=False) = limited model
# (model_only=False, zero_pad=True) = full sentics
@util.dump_log
def vectorize_collection(df, output, model, shaping, word_limit, sentics=False, model_only=False, zero_pad=False):
    vector_collection = []

    for index, row in tqdm(df.iterrows(), "Creating vectors", total=df.shape[0]):
        vectors = vectorize_document(row.content, model, word_limit, sentics, model_only, zero_pad)

        if shaping == "sequence":
            vector_collection.append(vectors)
        elif shaping == "avg":
            averaged = np.mean(vectors, axis=0)
            vector_collection.append(averaged)
        elif shaping == "avg_std":
            averaged = np.mean(vectors, axis=0)
            stddev = np.std(vectors, axis=0)
            combined = np.concatenate((averaged, stddev), axis=0)
            vector_collection.append(combined)

    with open(output, "wb") as outfile:
        pickle.dump(vector_collection, outfile)
        logging.info("Saved %s", output)
        # np.save(vector_collection, outfile, allow_pickle=False)

    return vector_collection


# TODO: pass model of None to not use word emebedding
# NOTE: you can also use sent_tokenize if needed
def vectorize_document(doc, model, word_limit=-1, sentics=False, model_only=False, zero_pad=False):
    # clean it
    doc = util.clean_symbols(util.clean_newlines(doc))

    # doc_words = doc.split(" ")
    doc = doc.lower()
    doc_words = word_tokenize(doc)

    # other cleaning:
    
    
    if word_limit == -1:
        word_limit = len(doc_words)

    if model is not None:
        doc_words = list(filter(lambda x: x in model.vocab, doc_words))  # TODO: use wordlimit
    
    if sentics:
        vectors = senticnetify.get_article_embedding(doc_words, model, model_only, zero_pad)
    else:    
        vectors = [model[word] for word in doc_words]

    return np.array(vectors)
