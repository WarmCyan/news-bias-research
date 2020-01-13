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


def find_tagged(tagged_words, pattern):
    found_patterns = []
    end = len(tagged_words) - len(pattern)
    for index, (word, pos) in enumerate(tagged_words, 1):

        if index == end:
            break

        i = 0
        found = True
        
        for part in pattern:
            if part != tagged_words[index + i][1]:
                found = False
                break
            i += 1

        if found:
            found_pattern = tagged_words[index:index + len(pattern)]
            found_patterns.append(found_pattern)

    return found_patterns
        

def get_sentiment_vector_instance(sentiment):
    pv = sentiment["polarity_value"]
    pi = sentiment["polarity_intense"]
    spl = sentiment["sentics"]["pleasentness"]
    satt = sentiment["sentics"]["attention"]
    ssen = sentiment["sentics"]["sensitivity"]
    sapt = sentiment["sentics"]["aptitude"]
    
    sentic_vector_instance = [pv, pi, spl, satt, ssen, sapt]
    return sentic_vector_instance


def get_article_embedding(doc_words, model):
    global sn

    tagged = []
    # sentences = sent_tokenize(doc_words)
    # for sentence in sentences:
        # tagged.extend()
    
    tagged = nltk.pos_tag(doc_words, tagset="universal")

    patterns = find_tagged(tagged, patterns[0])
    patterns.extend(find_tagged(tagged, patterns[1]))
    patterns.extend(find_tagged(tagged, patterns[2]))
    patterns.extend(find_tagged(tagged, patterns[3]))

    sentic_vectors = []
    
    for pattern in patterns: # change latter portion
        pattern_text = ""
        pattern_words = []
        for word in pattern:
            pattern_text += word[0]
            pattern_words.append(word[0])

        sentic_vector_instances = []
        
        try: 
            sentiment = sn.concept(pattern_text)
            sentic_vector_instances.append(get_sentiment_vector_instance(sentiment)
        except KeyError: 
            for word in pattern_words:
                try:
                    sentiment = sn.concept(word)
                    sentic_vector_instances.append(get_sentiment_vector_instance(sentiment)
                except KeyError: 
                    sentic_vector_instances.append([0,0,0,0,0,0])
                    continue
                

        if model is not None:
            for index, word in enumerate(pattern_words):
                vector = []
                if len(sentic_vector_instances) > 1:
                    vector.extend(sentic_vector_instances[index])
                else:
                    vector.extend(sentic_vector_instances[0])
                
                vector.extend(model[word])
                sentic_vectors.append(vector)
        else:
            sentic_vectors.extend(sentic_vector_instances)

    return sentic_vectors
