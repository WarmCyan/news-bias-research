import logging
import traceback
import nltk
import json
from senticnet.senticnet import SenticNet
import numpy as np


sn = SenticNet()


patterns = [
    ["NOUN"],
    ["NOUN", "NOUN"],
    ["VERB", "NOUN"],
    ["ADJ"],
    ["ADV"]
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
            pattern_tuple = (index, len(pattern), found_pattern)
            found_patterns.append(pattern_tuple)

    return found_patterns
        

def get_sentiment_vector_instance(sentiment):
    pv = sentiment["polarity_value"]
    pi = float(sentiment["polarity_intense"])
    spl = float(sentiment["sentics"]["pleasantness"])
    satt = float(sentiment["sentics"]["attention"])
    ssen = float(sentiment["sentics"]["sensitivity"])
    sapt = float(sentiment["sentics"]["aptitude"])

    if pv == "negative": 
        pv = -1.0
    else: 
        pv = 1.0
    
    sentic_vector_instance = [pv, pi, spl, satt, ssen, sapt]
    return sentic_vector_instance


# returns sentic_vector_instances
def get_sentics_for_pattern(pattern_text, pattern_words):
    global sn
    sentic_vector_instances = []

    try: 
        sentiment = sn.concept(pattern_text)
        sentic_vector_instances.append(get_sentiment_vector_instance(sentiment))
    except KeyError as e: 
        for word in pattern_words:
            try:
                sentiment = sn.concept(word)
                sentic_vector_instances.append(get_sentiment_vector_instance(sentiment))
            except KeyError: 
                sentic_vector_instances.append([0,0,0,0,0,0])
                continue

    return sentic_vector_instances


def get_pattern_text_and_words(pattern):
    pattern_text = ""
    pattern_words = []
    for word in pattern:
        pattern_text += word[0] + " "
        pattern_words.append(word[0])
    pattern_text = pattern_text.strip() # remove trailing space

    return pattern_text, pattern_words


# model_only of True means we're doing a limited embedding (no sentics involved, but only looking at the words that would be)
# zero_pad equals sentic_full
# (model_only=False, zero_pad=False) = normal sentic usage
# (model_only=True, zero_pad=False) = limited model
# (model_only=False, zero_pad=True) = full sentics
def get_article_embedding(doc_words, model, model_only=False, zero_pad=False):
    global sn
    global patterns

    tagged = []
    # sentences = sent_tokenize(doc_words)
    # for sentence in sentences:
        # tagged.extend()
    
    tagged = nltk.pos_tag(doc_words, tagset="universal")

    found_patterns = find_tagged(tagged, patterns[0])
    found_patterns.extend(find_tagged(tagged, patterns[1]))
    found_patterns.extend(find_tagged(tagged, patterns[2]))
    found_patterns.extend(find_tagged(tagged, patterns[3]))
    found_patterns.extend(find_tagged(tagged, patterns[4]))

    sentic_vectors = []

    # keep track of where any sentic available patterns start
    sentic_indices = []
    for index, length, pattern in found_patterns:
        sentic_indices.append(index)
    
    # if we're not doing full, just limited (either with or without sentics)
    if not zero_pad:
        for index, length, pattern in found_patterns: 
            pattern_text, pattern_words = get_pattern_text_and_words(pattern)

            # if we're actually including sentics and not just limiting w2v, get sentics
            sentic_vector_instances = []
            if not model_only:
                sentic_vector_instances = get_sentics_for_pattern(pattern_text, pattern_words)

            # get the model word embedding
            if model is not None:
                for index, word in enumerate(pattern_words):
                    vector = []
                    if not model_only:
                        if len(sentic_vector_instances) > 1:
                            vector.extend(sentic_vector_instances[index])
                        else:
                            vector.extend(sentic_vector_instances[0])
                    
                    vector.extend(model[word])
                    sentic_vectors.append(vector)
            else:
                sentic_vectors.extend(sentic_vector_instances)

        # make sure we have at least one vector of data for the one or two odd cases where something goes wrong and there's no valid words for the model
        if len(sentic_vectors) == 0:
            if model is not None:
                # sentic_vectors = [[0*306]] # this shouldn't have ever worked.........??
                if not model_only:
                    sentic_vectors = [[0]*306]
                else:
                    sentic_vectors = [[0]*300]
            else:
                # sentic_vectors = [[0*6]] #.....
                sentic_vectors = [[0]*6]
    else: # we're doing full, loop through every word (NOTE: will always use sentics if doing full, obviously, otherwise would just do only embedding)
        skip_next = False # used for two word patterns where we want to skip the next one (so we don't add it twice)

        in_pattern = 0
        not_in_pattern = 0
        
        for index, word in enumerate(doc_words):

            if skip_next:
                skip_next = False
                continue
            
            # check if we're actually in a pattern
            if index in sentic_indices:
                #print("In pattern")
                pattern_index, pattern_length, pattern = found_patterns[sentic_indices.index(index)]
                pattern_text, pattern_words = get_pattern_text_and_words(pattern)
                sentic_vector_instances = get_sentics_for_pattern(pattern_text, pattern_words)
                in_pattern += pattern_length
                #print("sentic vector instances")
                #print(sentic_vector_instances)

                # get the model word embedding
                if model is not None:
                    for index, word in enumerate(pattern_words):
                        vector = []
                        if len(sentic_vector_instances) > 1:
                            vector.extend(sentic_vector_instances[index])
                        else:
                            vector.extend(sentic_vector_instances[0])
                        
                        #print("Sentic vector only")
                        #print(vector)
                        vector.extend(model[word])
                        #print("Vector")
                        vector = np.asarray(vector)
                        #print(vector)
                        #print(vector.shape)
                        #if vector.shape != (306,):
                        #    print(pattern_text)
                        #    print(vector)
                        #    print(vector.shape) 
                        #    print("-----------")
                        sentic_vectors.append(vector)
                else:
                    sentic_vectors.extend(sentic_vector_instances)
                
                # don't re-do the next word!
                if pattern_length == 2:
                    skip_next = True
            else:
                not_in_pattern += 1
                vector = []

                # NOTE: while I'm inserting zeros, we technically could also look up this word in senticnet as well (just pass [word] as pattern)
                vector.extend([0]*6)
                vector.extend(model[word])
                sentic_vectors.append(np.asarray(vector))
                
        #print("IN PATTERN:",str(in_pattern))
        #print("NOT IN PATTERN:",str(not_in_pattern))

        sentic_vectors = np.asarray(sentic_vectors)
        #print(sentic_vectors.shape)

    return sentic_vectors
