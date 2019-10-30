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
def fill_source(selection_sources, problem, col_names, target_name, target_count, reject_minimum):
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
    
    for key in source_list:
        
        # reliable = selection_sources[problem][key]["reliable"]
        # unreliable = selection_sources[problem][key]["unreliable"]

        sets = []
        for name in col_names:
            sets.append(selection_sources[problem][key][name])

        if len(sets) == 2:
            _, meta = create_binary_selection("", sets[0], sets[1], target_name, count_per=target_count, reject_minimum=reject_minimum, overwrite=True, analysis_only=True)
        elif len(sets) == 3:
            _, meta = create_ternary_selection("", sets[0], sets[1], sets[2], target_name, count_per=target_count, reject_minimum=reject_minimum, overwrite=True, analysis_only=True)
            
        selection_sources[problem][key]["meta"] = meta

    return selection_sources
    

@util.dump_log
def create_selection_set_sources(target_count, reject_minimum, overwrite=False):
      
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
            "biased": {
                "mbfc": {},
                "as": {},
                "info": {}
                },
            "extreme_biased": {
                "mbfc": {},
                "as": {},
                "info": {}
                },
            "bias_direction": {
                "mbfc": {},
                "as": {},
                "info": {}
                },
            "extreme_bias_direction": {
                "mbfc": {},
                "as": {},
                "info": {}
                },
            "large_center_bias_direction": {
                "mbfc": {},
                "as": {},
                "info": {}
                }
            }

    selection_sources = fill_source(selection_sources, "reliability", ["reliable", "unreliable"], "reliable", target_count, reject_minimum)
    selection_sources = fill_source(selection_sources, "biased", ["biased", "unbiased"], "biased", target_count, reject_minimum)
    selection_sources = fill_source(selection_sources, "extreme_biased", ["biased", "unbiased"], "biased", target_count, reject_minimum)
    selection_sources = fill_source(selection_sources, "bias_direction", ["left", "center", "right"], "bias_direction", target_count, reject_minimum)
    selection_sources = fill_source(selection_sources, "extreme_bias_direction", ["left", "center", "right"], "bias_direction", target_count, reject_minimum)
    selection_sources = fill_source(selection_sources, "large_center_bias_direction", ["left", "center", "right"], "bias_direction", target_count, reject_minimum)

        
    with open("../data/cache/selection.json", 'w') as outfile:
        json.dump(selection_sources, outfile)
    return selection_sources

@util.dump_log
def get_selection_set(
    problem, source, count, random_seed, reject_minimum, overwrite
):
    labels_df = util.nela_load_labels()

    name = "selection_{0}_{1}_{2}_{3}".format(
        problem, source, count, random_seed
    )

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

        df, meta = create_binary_selection(
            name,
            reliable,
            unreliable,
            "reliable",
            count_per=count,
            reject_minimum=reject_minimum,
            overwrite=overwrite,
        )

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

        df, meta = create_binary_selection(
            name,
            biased,
            unbiased,
            "biased",
            count_per=count,
            reject_minimum=reject_minimum,
            overwrite=overwrite,
        )
        
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

        df, meta = create_binary_selection(
            name,
            biased,
            unbiased,
            "biased",
            count_per=count,
            reject_minimum=reject_minimum,
            overwrite=overwrite,
        )

    return df, name


@util.dump_log
def get_embedding_set(df, embedding_type, output_name, shaping, overwrite=False):
    logging.info("Creating %s embedding %s...", embedding_type, output_name)
    
    path = "../data/cache/" + output_name + "_" + embedding_type
    path_and_name = path + "/" + shaping + ".pkl"
    
    if not util.check_output_necessary(path_and_name, overwrite):
        # df = pickle.load(path_and_name)
        with open(path_and_name, 'rb') as infile:
            df = pickle.load(infile)
        return df
    
    try:
        os.mkdir(path)
    except:
        pass
    
    embedding_df = None
    if embedding_type == "w2v":
        embedding_df = word2vec_creator.run_w2v(df, path_and_name, shaping=shaping, word_limit=-1)
    elif embedding_type == "glove":
        embedding_df = word2vec_creator.run_glove(df, path_and_name, shaping=shaping, word_limit=-1)
    # TODO: fasttext

    return embedding_df


def clear_vector_model():
    word2vec_creator.clear_model()


def create_tfidf(df, output_name, max_features=10000, overwrite=False):
    logging.info("Creating tfidf %s...", output_name)

    path = "../data/cache/" + output_name

    if not util.check_output_necessary(path, overwrite):
        return

    try:
        os.mkdir(path)
    except:
        pass
    print(df.content[df.content.isnull()])

    corpus = list(df.content)
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)
    tfidf_final_vectors = tfidf_matrix.todense().tolist()

    with open(path + "/tfidf.pkl", "wb") as outfile:
        pickle.dump(tfidf_final_vectors, outfile)


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
        return df

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
        return df

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

    df_center[col_title] = 0
    df_left[col_title] = -1
    df_right[col_title] = 1

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


