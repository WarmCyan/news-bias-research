"""
Code to create, load, and manage datasets
"""

import pandas as pd
import tqdm as tqdm
import logging

import util


def create_binary_selection(output, positive_sources, negative_sources, col_title, count_per, reject_minimum=100, force_balance=True, random_seed=13, verbose=True, overwrite=False):

    if not util.check_output_necessary(output, overwrite): return
    
    df_positive, positive_counts, positive_rejected = random_balanced_sample(positive_sources, count=count_per, reject_minimum=reject_minimum, force_balance=force_balance, random_seed=random_seed, verbose=verbose)
    df_negative, negative_counts, negative_rejected = random_balanced_sample(negative_sources, count=count_per, reject_minimum=reject_minimum, force_balance=force_balance, random_seed=random_seed, verbose=verbose)

    df_negative[col_title] = 0
    df_positive[col_title] = 1

    df = util.stack_dfs(df_positive, df_negative)


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
        local_count = local_df.shape[0]
        if verbose:
            print(name, local_count, "articles")

        if local_count < reject_minimum:
            rejected.append(name)
            if verbose:
                print(name, "rejected")
            continue

        if local_count < minimum_count:
            minimum_count = local_count

        counts[name] = local_count
        building_df = util.stack_dfs(building_df, local_df)

    max_possible_balanced = minimum_count * len(counts.keys())
    if verbose:
        print(max_possible_balanced, "maximum possible balanced sample size")

    sample_df = None
    if count > max_possible_balanced:
        if verbose:
            print("Grabbing maximum")
        if force_balance:
            if verbose:
                print("Force balance requested, will not return", count, "as requested")

            for name in tqdm(counts.keys(), "Sampling", disable=(not verbose)):
                source_sample_df = building_df[building_df.source == name].sample(
                    minimum_count, random_state=random_seed
                )
                sample_df = util.stack_dfs(sample_df, source_sample_df)

        else:
            print("WARNING: unbalanced output")

            # TODO
    else:
        total_per = int(count / len(counts.keys()))
        remainder = count % len(counts.keys())
        if verbose:
            print("Grabbing", total_per, "per source")

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
