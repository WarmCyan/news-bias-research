"""
Code to create, load, and manage datasets
"""

import json
import logging
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import util
import word2vec_creator


def get_selection_sets():
    labels_df = util.nela_load_labels()

    # ----------------------------
    # OpenSources Reliability
    # ----------------------------

    os_unreliable_lbl = "Open Sources, unreliable"
    os_reliable_lbl = "Open Sources, reliable"

    os_unreliable = list(labels_df[labels_df[os_unreliable_lbl].notnull()].Source)
    os_reliable = list(labels_df[labels_df[os_reliable_lbl].notnull()].Source)

    df_os_reliability = create_binary_selection(
        "os_reliability.csv",
        os_reliable,
        os_unreliable,
        "reliable",
        count_per=4676,
        reject_minimum=300,
    )

    # ----------------------------
    # Media Bias / Fact Check Reliability
    # ----------------------------

    mbfc_lbl = "Media Bias / Fact Check, factual_reporting"

    mbfc_reliable = list(labels_df[labels_df[mbfc_lbl] > 3].Source)
    mbfc_unreliable = list(labels_df[labels_df[mbfc_lbl] <= 3].Source)

    df_mbfc_reliability = create_binary_selection(
        "mbfc_reliability.csv",
        mbfc_reliable,
        mbfc_unreliable,
        "reliable",
        count_per=5000,
        reject_minimum=300,
    )

    # ----------------------------
    # NewsGuard Reliability
    # ----------------------------

    ng_lbl = "NewsGuard, overall_class"

    ng_reliable = list(labels_df[labels_df[ng_lbl] == 1.0].Source)
    ng_unreliable = list(labels_df[labels_df[ng_lbl] == 0.0].Source)

    df_ng_reliability = create_binary_selection(
        "ng_reliability.csv",
        ng_reliable,
        ng_unreliable,
        "reliable",
        count_per=5000,
        reject_minimum=300,
    )

    return df_os_reliability, df_mbfc_reliability, df_ng_reliability


def create_word2vec(df, output_name, overwrite=False):
    logging.info("Creating word2vec %s...", output_name)

    path = "../data/cache/" + output_name

    if not util.check_output_necessary(path, overwrite):
        return

    try:
        os.mkdir(path)
    except:
        pass

    word2vec_creator.run_w2v(df, path + "/sequence", shaping="sequence", word_limit=-1)
    word2vec_creator.run_w2v(df, path + "/avg", shaping="avg", word_limit=-1)


def create_glove(df, output_name, overwrite=False):
    logging.info("Creating glove %s...", output_name)

    path = "../data/cache/" + output_name

    if not util.check_output_necessary(path, overwrite):
        return

    try:
        os.mkdir(path)
    except:
        pass

    word2vec_creator.run_glove(
        df, path + "/sequence", shaping="sequence", word_limit=-1
    )
    word2vec_creator.run_glove(df, path + "/avg", shaping="avg", word_limit=-1)


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
):
    logging.info("Creating binary selection %s...", output)

    path = "../data/cache/" + output

    if not util.check_output_necessary(path, overwrite):
        logging.info("Loading %s...", path)
        df = pd.read_csv(path)
        return df

    df_positive, positive_counts, positive_rejected = random_balanced_sample(
        positive_sources,
        count=count_per,
        reject_minimum=reject_minimum,
        force_balance=force_balance,
        random_seed=random_seed,
        verbose=verbose,
    )
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

    logging.info("Caching...")
    df.to_csv(path)
    meta = {
        "postive_counts": positive_counts,
        "negative_counts": negative_counts,
        "positive_rejected": positive_rejected,
        "negative_rejected": negative_rejected,
    }

    with open(path + "_meta.json", "w") as outfile:
        json.dump(meta, outfile)

    return df


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
        local_df = local_df[local_df.content.notnull()]
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

    return sample_df, returned_counts, rejected
