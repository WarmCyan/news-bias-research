import logging
import pickle

import gensim
from gensim.models import fasttext
import numpy as np
from tqdm import tqdm

import util

model = None


# shaping:
#   - sequence (keep separated)
#   - avg (average the w2v)
# TODO: add turning into square or flattening
def run_w2v(df, output, shaping="sequence", word_limit=-1):
    global model

    if model is None:
        logging.info("Loading word2vec model...")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../data/raw/models/GoogleNews-vectors-negative300.bin", binary=True
        )

    return vectorize_collection(df, output, model, shaping, word_limit)


def run_glove(df, output, shaping="sequence", word_limit=-1):
    global model

    if model is None:
        logging.info("Loading glove model...")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            "../data/cache/models/glove2word2vec_pretrained.model"
        )

    return vectorize_collection(df, output, model, shaping, word_limit)


def run_fasttext(df, output, shaping="sequence", word_limit=-1):
    global model

    if model is None:
        logging.info("Loading fasttext model...")

        model = gensim.models.fasttext.load_facebook_vectors('../data/raw/models/wiki.en.bin')

    return vectorize_collection(df, output, model, shaping, word_limit)


def clear_model():
    global model
    del model
    model = None


def vectorize_collection(df, output, model, shaping, word_limit):
    vector_collection = []

    for index, row in tqdm(df.iterrows(), "Creating vectors", total=df.shape[0]):
        vectors = vectorize_document(row.content, model, word_limit)

        if shaping == "sequence":
            vector_collection.append(vectors)
        elif shaping == "avg":
            averaged = np.mean(vectors, axis=0)  # TODO: verify axis is correct
            vector_collection.append(averaged)

    with open(output, "wb") as outfile:
        pickle.dump(vector_collection, outfile)
        logging.info("Saved %s", output)
        # np.save(vector_collection, outfile, allow_pickle=False)

    return vector_collection


def vectorize_document(doc, model, word_limit=-1):
    # clean it
    doc = util.clean_symbols(util.clean_newlines(doc))

    doc_words = doc.split(" ")
    if word_limit == -1:
        word_limit = len(doc_words)
    doc_words = filter(lambda x: x in model.vocab, doc_words)  # TODO: use wordlimit
    vectors = [model[word] for word in doc_words]

    return np.array(vectors)
