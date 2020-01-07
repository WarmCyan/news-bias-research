import nltk
import json
from senticnet.senticnet import SenticNet


sn = SenticNet()


patterns = [
    ["NOUN"]
    ["NOUN", "NOUN"]
    ["VERB", "NOUN"],
    ["ADJ"]
]



def find_tagged(tagged_words):
    for index, (word, pos) in enumerate(tagged_words, 1):
        pass       

def get_article_embedding(doc_words, model):
    global sn

    tagged = []
    # sentences = sent_tokenize(doc_words)
    # for sentence in sentences:
        # tagged.extend()
    
    tagged = nltk.pos_tag(doc_words, tagset="universal")

    sentic_vectors = []
    
    # TODO: find patterns and words to check with senticnet

    for pattern in tagged: # change latter portion
        pattern_text = ""
        pattern_words = []
        for word in pattern:
            pattern_text += word[0]
            pattern_words.append(word[0])
        try: 
            sentiment = sn.concept(pattern_text)
        except KeyError: 
            # if not, try individual words in pattern
            continue
        
        pv = sentiment["polarity_value"]
        pi = sentiment["polarity_intense"]
        spl = sentiment["sentics"]["pleasentness"]
        satt = sentiment["sentics"]["attention"]
        ssen = sentiment["sentics"]["sensitivity"]
        sapt = sentiment["sentics"]["aptitude"]

        sentic_vector = [pv, pi, spl, satt, ssen, sapt]

        if model is not None:
            for word in pattern_words:
                vector = []
                vector.extend(sentic_vector)
                vector.extend(model[word])
                sentic_vectors.append(vector)
        else:
            sentic_vectors.append(sentic_vector)

    return sentic_vectors
