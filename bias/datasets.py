"""
Code to create, load, and manage datasets
"""

import json
import logging
import os
import pickle
import copy

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import util
import word2vec_creator


tfidf_vocab = None

# returns the equivalent of a selection set
def load_articlelevel_set(binary=True, bias=True, threshold=8.4):
    df = util.load_scraped_mpc()
    df = df[df.content.notnull()]
    df = df[df.content.str.split().apply(len) > 50]
    
    if bias:
        threshold=8.4
        df_left = df[df.Bias < -threshold]
        df_right = df[df.Bias > threshold]
        df_center = df[(df.Bias >= -threshold) & (df.Bias <= threshold)]

        if binary:
            df_left["biased"] = 1
            df_right["biased"] = 1
            df_center["biased"] = 0
        else:
            df_left["biased"] = 0
            df_right["biased"] = 2
            df_center["biased"] = 1

        df_final = util.stack_dfs(df_left, df_right)
        df_final = util.stack_dfs(df_final, df_center)
    else:
        df_reliable = df[df.Quality > 32]
        df_unreliable = df[df.Quality <= 32]

        df_reliable["reliable"] = 1
        df_unreliable["reliable"] = 0
        df_final = util.stack_dfs(df_reliable, df_unreliable)

    return df_final

# NOTE: for reliability, assumes selection.json already exists? (and that folds exist)
def load_fold(n, count_per, selection_tag="", binary=True, bias=True, overwrite=False):
    fold_sources = util.load_fold_divisions_dataset(selection_tag, bias)
    fold = fold_sources[n]

    if binary:

        if bias:
            positive_sources = fold["left"]
            positive_sources.extend(fold["right"])
            
            negative_sources = fold["center"]

            df, meta = create_binary_selection(
                f"{selection_tag}binary_fold_" + str(count_per) + "_" + str(n), 
                positive_sources,
                negative_sources,
                "biased",
                count_per=count_per,
                reject_minimum=300,
                random_seed=13,
                overwrite=overwrite
            )
        else:
            reliable_sources = fold["reliable"]
            unreliable_sources = fold["unreliable"]

            df, meta = create_binary_selection(
                f"reliable_{selection_tag}binary_fold_" + str(count_per) + "_" + str(n), 
                reliable_sources,
                unreliable_sources,
                "reliable",
                count_per=count_per,
                reject_minimum=300,
                random_seed=13,
                overwrite=overwrite
            )
    else:
        df, meta = create_ternary_selection(
            f"{selection_tag}ternary_fold_" + str(count_per) + "_" + str(n),
            fold["left"],
            fold["center"],
            fold["right"],
            "biased",
            count_per=count_per,
            reject_minimum=300,
            force_balance=True,
            random_seed=13,
            overwrite=overwrite
        )

    return df, meta


@util.dump_log
def load_folds(n, count_per, selection_tag="", binary=True, bias=True, overwrite=False):
    fold_sources = util.load_fold_divisions_dataset(selection_tag)

    combined = None

    fold_indices = [x for x in range(0, 10) if x != n]
    for fold_index in fold_indices:
        df, _ = load_fold(fold_index, count_per, selection_tag, binary, bias, overwrite)   
        combined = util.stack_dfs(combined, df)

    test, _ = load_fold(n, count_per, selection_tag, binary, bias, overwrite)

    return combined, test

def list_difference(x, y):
    return [item for item in x if item not in y]


# expects list of dictionaries, where each dict elem is a labeled list
# (also assumes each one is a deep copy already)
# NOTE: alters all passed lists in place
def remove_conflicts(list_sets, source_names):
    list_names = list_sets[0].keys()

    to_remove = []

    # yikes......
    # (for each source), make sure that source isn't in a different labeled list in any of the other sets
    for list_set in list_sets:
        for name in list_names:
            for source in list_set[name]:
                for other_list_set in list_difference(list_sets, [list_set]):
                    for other_name in list_difference(list_names, [name]):
                        if source in other_list_set[other_name]:
                            to_remove.append(source)

    to_remove = list(set(to_remove))
    removed = []

    for item in to_remove:
        source_index = 0
        for list_set in list_sets:
            for name in list_names:
                try:
                    list_set[name].remove(item)
                    logging.info("%s conflict removed from %s %s", item, source_names[source_index], name)
                    removed.append((item, source_names[source_index], name))
                except:
                    logging.error("Failed to remove conflict")
                    pass
            source_index += 1

    logging.info("%i conflicts found in set %s", len(to_remove), str(list_names))

    return to_remove, removed


@util.dump_log
def get_original_selection_set(problem, source):
    labels_df = util.nela_load_labels()
    if problem == "reliability":
        if source == "os":
            reliable_lbl = "Open Sources, reliable"
            unreliable_lbl = "Open Sources, unreliable"

            unreliable = list(labels_df[labels_df[unreliable_lbl].notnull()].Source)
            reliable = list(labels_df[labels_df[reliable_lbl].notnull()].Source)

        if source == "mbfc":
            lbl = "Media Bias / Fact Check, factual_reporting"

            reliable = list(labels_df[labels_df[lbl] > 3].Source)
            unreliable = list(labels_df[labels_df[lbl] <= 3].Source)

        if source == "ng":
            ng_lbl = "NewsGuard, overall_class"
            
            reliable = list(labels_df[labels_df[ng_lbl] == 1.0].Source)
            unreliable = list(labels_df[labels_df[ng_lbl] == 0.0].Source)

        return [reliable, unreliable]
    elif problem == "biased":
        if source == "mbfc":
            mbfc_lbl = "Media Bias / Fact Check, label"
            
            biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "left_bias")
                    | (labels_df[mbfc_lbl] == "left_center_bias")
                    | (labels_df[mbfc_lbl] == "right_center_bias")
                    | (labels_df[mbfc_lbl] == "right_bias")
                ].Source
            )

            unbiased = list(
                labels_df[labels_df[mbfc_lbl] == "least_biased"].Source
            )
        
        elif source == "as":
            as_lbl = "Allsides, bias_rating"
            biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Left")
                    | (labels_df[as_lbl] == "Right")
                    | (labels_df[as_lbl] == "Lean Left")
                    | (labels_df[as_lbl] == "Lean Right")
                ].Source
            )
            unbiased = list(labels_df[labels_df[as_lbl] == "Center"].Source)

        return [biased, unbiased]
    elif problem == "extreme_biased":
        if source == "mbfc":
            mbfc_lbl = "Media Bias / Fact Check, label"
            
            biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "left_bias")
                    | (labels_df[mbfc_lbl] == "right_bias")
                ].Source
            )

            unbiased = list(
                labels_df[labels_df[mbfc_lbl] == "least_biased"].Source
            )
        
        elif source == "as":
            as_lbl = "Allsides, bias_rating"
            biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Left")
                    | (labels_df[as_lbl] == "Right")
                ].Source
            )
            unbiased = list(labels_df[labels_df[as_lbl] == "Center"].Source)

        return [biased, unbiased]
    elif problem == "bias_direction":
        if source == "mbfc":
            mbfc_lbl = "Media Bias / Fact Check, label"
            
            left_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "left_bias")
                    | (labels_df[mbfc_lbl] == "left_center_bias")
                ].Source
            )
            
            right_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "right_bias")
                    | (labels_df[mbfc_lbl] == "right_center_bias")
                ].Source
            )

            center_biased = list(
                labels_df[labels_df[mbfc_lbl] == "least_biased"].Source
            )
        
        elif source == "as":
            as_lbl = "Allsides, bias_rating"
            left_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Left")
                    | (labels_df[as_lbl] == "Lean Left")
                ].Source
            )
            right_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Right")
                    | (labels_df[as_lbl] == "Lean Right")
                ].Source
            )
            center_biased = list(labels_df[labels_df[as_lbl] == "Center"].Source)

        return [left_biased, center_biased, right_biased]
    
    elif problem == "extreme_bias_direction":
        if source == "mbfc":
            mbfc_lbl = "Media Bias / Fact Check, label"
            
            left_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "left_bias")
                ].Source
            )
            
            right_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "right_bias")
                ].Source
            )

            center_biased = list(
                labels_df[labels_df[mbfc_lbl] == "least_biased"].Source
            )
        
        elif source == "as":
            as_lbl = "Allsides, bias_rating"
            left_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Left")
                ].Source
            )
            right_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Right")
                ].Source
            )
            center_biased = list(labels_df[labels_df[as_lbl] == "Center"].Source)

        return [left_biased, center_biased, right_biased]
    elif problem == "large_center_bias_direction":
        if source == "mbfc":
            mbfc_lbl = "Media Bias / Fact Check, label"
            
            left_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "left_bias")
                ].Source
            )
            
            right_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "right_bias")
                ].Source
            )
            center_biased = list(
                labels_df[
                    (labels_df[mbfc_lbl] == "right_center_bias")
                    | (labels_df[mbfc_lbl] == "left_center_bias")
                    | (labels_df[mbfc_lbl] == "least_biased")
                ].Source
            )
        elif source == "as":
            as_lbl = "Allsides, bias_rating"
            left_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Left")
                ].Source
            )
            right_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Right")
                ].Source
            )
            center_biased = list(
                labels_df[
                    (labels_df[as_lbl] == "Center")
                    | (labels_df[as_lbl] == "Lean Right")
                    | (labels_df[as_lbl] == "Lean Left")
                ].Source
            )

        return [left_biased, center_biased, right_biased]


@util.dump_log
def fill_source(selection_sources, problem, col_names, target_name, target_count, reject_minimum, verbose):
    to_fix = []

    source_list = list_difference(selection_sources[problem].keys(), ["info"])

    for key in source_list:
        fix_add = {}

        sets = get_original_selection_set(problem, key)
        for i in range(0, len(sets)):
            set_name = col_names[i]
            selection_sources[problem][key][set_name + "_original"] = sets[i]

            set_copy = copy.deepcopy(sets[i])
            selection_sources[problem][key][set_name] = set_copy
            fix_add[set_name] = set_copy
            
        to_fix.append(fix_add)
            
    conflicts, removed = remove_conflicts(to_fix, source_list)
    selection_sources[problem]["info"]["conflicts"] = conflicts
    selection_sources[problem]["info"]["removed"] = removed
    selection_sources[problem]["info"]["target_name"] = target_name
    selection_sources[problem]["info"]["col_names"] = col_names
    
    for key in source_list:
        
        # reliable = selection_sources[problem][key]["reliable"]
        # unreliable = selection_sources[problem][key]["unreliable"]

        sets = []
        for name in col_names:
            sets.append(selection_sources[problem][key][name])

        if len(sets) == 2:
            _, meta = create_binary_selection("", sets[0], sets[1], target_name, count_per=target_count, reject_minimum=reject_minimum, overwrite=True, analysis_only=True, verbose=verbose)
        elif len(sets) == 3:
            _, meta = create_ternary_selection("", sets[0], sets[1], sets[2], target_name, count_per=target_count, reject_minimum=reject_minimum, overwrite=True, analysis_only=True, verbose=verbose)
            
        selection_sources[problem][key]["meta"] = meta

    # get the final source differences to make testing sets
    for key in source_list:
        for col in col_names:
            selection_sources[problem][key][col + "_test"] = {}
            for other_key in list_difference(source_list, [key]):
                testing_sources = list_difference(selection_sources[problem][other_key][col], selection_sources[problem][key][col])
                selection_sources[problem][key][col + "_test"][other_key] = testing_sources

    return selection_sources
    

@util.dump_log
def create_selection_set_sources(target_count, reject_minimum, overwrite=False, verbose=True):
      
    path = "../data/cache/selection.json"
    if not util.check_output_necessary(path, overwrite):
        with open(path, 'r') as infile:
            return json.load(infile)

    selection_sources = {
            "reliability": {
                "os": {},
                "mbfc": {},
                "ng": {},
                "info": {}
                },
            # "biased": {
            #     "mbfc": {},
            #     "as": {},
            #     "info": {}
            #     },
            # "extreme_biased": {
            #     "mbfc": {},
            #     "as": {},
            #     "info": {}
            #     },
            # "bias_direction": {
            #     "mbfc": {},
            #     "as": {},
            #     "info": {}
            #     },
            # "extreme_bias_direction": {
            #     "mbfc": {},
            #     "as": {},
            #     "info": {}
            #     },
            # "large_center_bias_direction": {
            #     "mbfc": {},
            #     "as": {},
            #     "info": {}
            #     }
            }

    selection_sources = fill_source(selection_sources, "reliability", ["reliable", "unreliable"], "reliable", target_count, reject_minimum, verbose)
    #selection_sources = fill_source(selection_sources, "biased", ["biased", "unbiased"], "biased", target_count, reject_minimum, verbose)
    #selection_sources = fill_source(selection_sources, "extreme_biased", ["biased", "unbiased"], "biased", target_count, reject_minimum, verbose)
    
    # selection_sources = fill_source(selection_sources, "bias_direction", ["left", "center", "right"], "bias_direction", target_count, reject_minimum)
    # selection_sources = fill_source(selection_sources, "extreme_bias_direction", ["left", "center", "right"], "bias_direction", target_count, reject_minimum)
    # selection_sources = fill_source(selection_sources, "large_center_bias_direction", ["left", "center", "right"], "bias_direction", target_count, reject_minimum)

        
    with open("../data/cache/selection.json", 'w') as outfile:
        json.dump(selection_sources, outfile)
    return selection_sources

# can't run on cluster (if not already created)
@util.dump_log
def get_selection_set(
    problem, source, count, random_seed, reject_minimum, overwrite, verbose
):
    labels_df = util.nela_load_labels()

    selection_sources = create_selection_set_sources(count, reject_minimum, False, verbose)

    name = "selection_{0}_{1}_{2}".format(
        problem, source, random_seed
    )

    if problem == "reliability" or problem == "biased" or problem == "extreme_biased":
        list_pos_name = selection_sources[problem]["info"]["col_names"][0]
        list_neg_name = selection_sources[problem]["info"]["col_names"][1]
        
        list_pos = selection_sources[problem][source][list_pos_name]
        list_neg = selection_sources[problem][source][list_neg_name]
        
        df, meta = create_binary_selection(
            name,
            list_pos,
            list_neg,
            selection_sources[problem]["info"]["target_name"],
            count_per=selection_sources[problem][source]["meta"]["count_per"],
            reject_minimum=reject_minimum,
            overwrite=overwrite,
            verbose=verbose,
        )

    return df, name


@util.dump_log
def get_embedding_set(df, embedding_type, output_name, shaping, selection_tag="", overwrite=False):
    logging.info("Creating %s embedding %s...", embedding_type, output_name)
    
    path = "../data/cache/" + selection_tag + output_name + "_" + embedding_type
    path_and_name = path + "/" + shaping + ".pkl.pkl"
    
    if not util.check_output_necessary(path_and_name, overwrite):
        # df = pickle.load(path_and_name)
        with open(path_and_name, 'rb') as infile:
            df = pickle.load(infile)
        return df
    
    util.create_dir(path)
    
    embedding_df = None
    logging.debug("Preparing to run vectorization")
    if embedding_type == "w2v":
        embedding_df = word2vec_creator.run_w2v(df, path_and_name, shaping=shaping, word_limit=-1, sentics=False)
    elif embedding_type == "w2v_sentic":
        embedding_df = word2vec_creator.run_w2v(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True)
    elif embedding_type == "w2v_sentic_full":
        embedding_df = word2vec_creator.run_w2v(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True, zero_pad=True)
    elif embedding_type == "w2v_limit":
        # NOTE: yes, it looks like sentics should be false, but this is how we limit w2v embeddings to just the patterns the sentics would have been calculated on
        embedding_df = word2vec_creator.run_w2v(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True, model_only=True)
    elif embedding_type == "glove":
        embedding_df = word2vec_creator.run_glove(df, path_and_name, shaping=shaping, word_limit=-1, sentics=False)
    elif embedding_type == "glove_sentic":
        embedding_df = word2vec_creator.run_glove(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True)
    elif embedding_type == "glove_sentic_full":
        embedding_df = word2vec_creator.run_glove(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True, zero_pad=True)
    elif embedding_type == "glove_limit":
        # NOTE: yes, it looks like sentics should be false, but this is how we limit glove embeddings to just the patterns the sentics would have been calculated on
        embedding_df = word2vec_creator.run_glove(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True, model_only=True)
    elif embedding_type == "fasttext":
        embedding_df = word2vec_creator.run_fasttext(df, path_and_name, shaping=shaping, word_limit=-1, sentics=False)
    elif embedding_type == "fasttext_sentic":
        embedding_df = word2vec_creator.run_fasttext(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True)
    elif embedding_type == "fasttext_sentic_full":
        embedding_df = word2vec_creator.run_fasttext(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True, zero_pad=True)
    elif embedding_type == "fasttext_limit":
        # NOTE: yes, it looks like sentics should be false, but this is how we limit fasttext embeddings to just the patterns the sentics would have been calculated on
        embedding_df = word2vec_creator.run_fasttext(df, path_and_name, shaping=shaping, word_limit=-1, sentics=True, model_only=True)
    elif embedding_type == "sentic":
        embedding_df = word2vec_creator.vectorize_collection(df, path_and_name, model=None, shaping=shaping, word_limit=-1, sentics=True)
    elif embedding_type == "tfidf":
        # TODO: add ability to add sentics here too?
        embedding_df = create_tfidf(df, path, overwrite=True)
        # NOTE: would have to save it (and check for preexisting) separately as well then
        

    return embedding_df




def clear_vector_model():
    word2vec_creator.clear_model()


def create_tfidf(df, path, max_features=5000, overwrite=False):
    global tfidf_vocab

    #logging.info("Creating tfidf %s...", path)

    #if not util.check_output_necessary(path + "/tfidf.pkl.pkl", overwrite):
    #    return

    try:
        os.mkdir(path)
    except:
        logging.error("Failed to create tfidf directory")
        pass
    print(df.content[df.content.isnull()])

    corpus = list(df.content)
    vectorizer = TfidfVectorizer(max_features=max_features, vocabulary=tfidf_vocab)

    vectorizer.fit(corpus)
    
    if tfidf_vocab is None:
        tfidf_vocab = vectorizer.vocabulary_
    
    tfidf_matrix = vectorizer.transform(corpus)
    tfidf_final_vectors = tfidf_matrix.todense().tolist()

    with open(path + "/tfidf.pkl.pkl", "wb") as outfile:
        pickle.dump(tfidf_final_vectors, outfile)

    return tfidf_final_vectors

def create_binary_selection(
    output,
    positive_sources,
    negative_sources,
    col_title,
    count_per,
    reject_minimum=100,
    force_balance=True,
    random_seed=13,
    verbose=True,
    overwrite=False,
    analysis_only=False
):
    logging.info("Creating binary selection %s...", output)

    path = "../data/cache/" + output

    if not util.check_output_necessary(path, overwrite):
        logging.info("Loading %s...", path)
        # df = pd.read_csv(path)
        df = pd.read_pickle(path)
        #df = pd.read_csv(path)
        print(df[df.content.isnull()])
        return df, None

    df_positive, positive_counts, positive_rejected = random_balanced_sample(
        positive_sources,
        count=count_per,
        reject_minimum=reject_minimum,
        force_balance=force_balance,
        random_seed=random_seed,
        verbose=verbose,
    )

    positive_max = df_positive.shape[0]
    if positive_max < count_per and force_balance:
        count_per = positive_max

    df_negative, negative_counts, negative_rejected = random_balanced_sample(
        negative_sources,
        count=count_per,
        reject_minimum=reject_minimum,
        force_balance=force_balance,
        random_seed=random_seed,
        verbose=verbose,
    )

    # if too much let's reget positive so that they are balanced
    negative_max = df_negative.shape[0]
    if negative_max < count_per and force_balance:
        logging.info("Imbalanced, regetting positive...")
        count_per = negative_max
        df_positive, positive_counts, positive_rejected = random_balanced_sample(
            positive_sources,
            count=count_per,
            reject_minimum=reject_minimum,
            force_balance=force_balance,
            random_seed=random_seed,
            verbose=verbose,
        )

        # NOTE: this shouldn't be necessary, but I keep losing my negative counts...
        df_negative, negative_counts, negative_rejected = random_balanced_sample(
            negative_sources,
            count=count_per,
            reject_minimum=reject_minimum,
            force_balance=force_balance,
            random_seed=random_seed,
            verbose=verbose,
        )

    df_negative[col_title] = 0
    df_positive[col_title] = 1

    df = util.stack_dfs(df_positive, df_negative)

    if not analysis_only:
        logging.info("Caching...")
        df.to_pickle(path, compression=None)
    # df.to_csv(path, encoding="utf-8")
    meta = {
        "count_per": count_per,
        "postive_counts": positive_counts,
        "negative_counts": negative_counts,
        "positive_rejected": positive_rejected,
        "negative_rejected": negative_rejected,
    }

    if not analysis_only:
        with open(path + "_meta.json", "w") as outfile:
            json.dump(meta, outfile)

    return df, meta

# can't run on cluster
def random_balanced_sample(
    source_name_array,
    count,
    reject_minimum=100,
    force_balance=True,
    random_seed=13,
    verbose=True,
):
    building_df = None
    counts = {}
    minimum_count = 100000000  # derp
    returned_counts = {}
    rejected = []

    for name in tqdm(source_name_array, "Querying sources", disable=(not verbose)):
        local_df = util.nela_load_articles_from_source(name)
        local_df = local_df[(local_df.content.notnull()) & (local_df.content != "")]
        local_df = local_df[local_df.content.str.split().apply(len) > 50] # filter out anything under 50 words
        local_count = local_df.shape[0]
        if verbose:
            tqdm.write("{0} {1} articles".format(name, local_count))
            # logging.info("%s %s articles", name, local_count)

        if local_count < reject_minimum:
            rejected.append(name)
            if verbose:
                tqdm.write(name + " rejected")
                # logging.info("%s rejected", name)
            continue

        if local_count < minimum_count:
            minimum_count = local_count

        counts[name] = local_count
        building_df = util.stack_dfs(building_df, local_df)


    # Preliminary filtering - remove any articles under 50 words
    #building_df = building_df[building_df.content.str.split().apply(len) > 50]
    #logging.debug("Remaining article count after filter: %s", str(building_df.shape[0]))
        

    max_possible_balanced = minimum_count * len(counts.keys())
    if verbose:
        tqdm.write(
            str(max_possible_balanced) + " maximum possible balanced sample size"
        )
        # logging.info("%s maximum possible balanced sample size", max_possible_balanced)

    sample_df = None
    if count > max_possible_balanced:
        if verbose:
            tqdm.write("Grabbing maximum")
            # logging.info("Grabbing maximum")
        if force_balance:
            if verbose:
                # logging.info(
                #     "Force balance requested, will not return %s as requested", count
                # )
                pass

            for name in tqdm(counts.keys(), "Sampling", disable=(not verbose)):
                source_sample_df = building_df[building_df.source == name].sample(
                    minimum_count, random_state=random_seed
                )
                returned_counts[name] = source_sample_df.shape[0]
                sample_df = util.stack_dfs(sample_df, source_sample_df)

        else:
            logging.warn("WARNING: unbalanced output")

            # TODO
    else:
        total_per = int(count / len(counts.keys()))
        remainder = count % len(counts.keys())
        if verbose:
            tqdm.write("Grabbing " + str(total_per) + " per source")
            # logging.info("Grabbing %s per source", total_per)

        for name in tqdm(counts.keys(), "Sampling", disable=(not verbose)):
            sample_size = total_per
            if remainder > 0:
                sample_size += 1
                remainder -= 1
            #logging.debug("Size of %s df: %s", name, building_df[building_df.source == name].shape[0])
            source_sample_df = building_df[building_df.source == name].sample(
                sample_size, random_state=random_seed
            )
            returned_counts[name] = source_sample_df.shape[0]
            sample_df = util.stack_dfs(sample_df, source_sample_df)

    print(sample_df.content[sample_df.content.isnull()])
    return sample_df, copy.deepcopy(returned_counts), copy.deepcopy(rejected)


def create_ternary_selection(
    output,
    left_sources,
    center_sources,
    right_sources,
    col_title,
    count_per,
    reject_minimum=100,
    force_balance=True,
    random_seed=13,
    verbose=True,
    overwrite=False,
    analysis_only=False
):
    logging.info("Creating ternary selection %s...", output)

    path = "../data/cache/" + output

    if not util.check_output_necessary(path, overwrite):
        logging.info("Loading %s...", path)
        df = pd.read_pickle(path)
        print(df[df.content.isnull()])
        return df, None

    df_left, left_counts, left_rejected = random_balanced_sample(
        left_sources,
        count=count_per,
        reject_minimum=reject_minimum,
        force_balance=force_balance,
        random_seed=random_seed,
        verbose=verbose,
    )

    left_max = df_left.shape[0]
    if left_max < count_per and force_balance:
        count_per = left_max

    df_center, center_counts, center_rejected = random_balanced_sample(
        center_sources,
        count=count_per,
        reject_minimum=reject_minimum,
        force_balance=force_balance,
        random_seed=random_seed,
        verbose=verbose,
    )

    # if too much let's reget positive so that they are balanced
    center_max = df_center.shape[0]
    if center_max < count_per and force_balance:
        count_per = center_max
        
    # right gauranteed to have <= center (max_per)
    df_right, right_counts, right_rejected = random_balanced_sample(
        right_sources,
        count=count_per,
        reject_minimum=reject_minimum,
        force_balance=force_balance,
        random_seed=random_seed,
        verbose=verbose,
    )

    right_max = df_right.shape[0]

    if left_max > count_per or center_max > right_max:
        count_per = min(left_max, right_max, center_max)
        logging.info("Imbalanced, regetting positive...")
        df_left, left_counts, left_rejected = random_balanced_sample(
            left_sources,
            count=count_per,
            reject_minimum=reject_minimum,
            force_balance=force_balance,
            random_seed=random_seed,
            verbose=verbose,
        )
        df_center, center_counts, center_rejected = random_balanced_sample(
            center_sources,
            count=count_per,
            reject_minimum=reject_minimum,
            force_balance=force_balance,
            random_seed=random_seed,
            verbose=verbose,
        )
        df_right, right_counts, right_rejected = random_balanced_sample(
            right_sources,
            count=count_per,
            reject_minimum=reject_minimum,
            force_balance=force_balance,
            random_seed=random_seed,
            verbose=verbose,
        )

    df_center[col_title] = 1
    df_left[col_title] = 0
    df_right[col_title] = 2

    df = util.stack_dfs(df_left, df_center)
    df = util.stack_dfs(df, df_right)

    if not analysis_only:
        logging.info("Caching...")
        df.to_pickle(path, compression=None)
    # df.to_csv(path, encoding="utf-8")
    meta = {
        "count_per": count_per,
        "left_counts": left_counts,
        "center_counts": center_counts,
        "right_counts": right_counts,
        "left_rejected": left_rejected,
        "center_rejected": center_rejected,
        "right_rejected": right_rejected,
    }

    if not analysis_only:
        with open(path + "_meta.json", "w") as outfile:
            json.dump(meta, outfile)

    return df, meta


