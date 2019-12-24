import nltk
import json
from senticnet.senticnet import SenticNet


sn = SenticNet()


# TODO: maybe pass in model here from w2v_creator vectorize_document?
def get_article_embedding(doc_words):
    global sn
    
    tagged = nltk.pos_tag(doc_words)
    
    # TODO: find patterns and words to check with senticnet

    for pattern in tagged: # change latter portion
        try: 
            sentiment = sn.concept(pattern)
        except KeyError: 
            continue
        
        pv = sentiment["polarity_value"]
        pi = sentiment["polarity_intense"]
        spl = sentiment["sentics"]["pleasentness"]
        satt = sentiment["sentics"]["attention"]
        ssen = sentiment["sentics"]["sensitivity"]
        sapt = sentiment["sentics"]["aptitude"]
