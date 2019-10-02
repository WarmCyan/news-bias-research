""" Utility functions for jupyter notebooks """

import os

import pandas as pd

DATA_RAW_PATH = "../data/raw/"

GT_COLS = {
    "NewsGuard": [
        "NewsGuard, Does not repeatedly publish false content",
        "NewsGuard, Gathers and presents information responsibly",
        "NewsGuard, Regularly corrects or clarifies errors",
        "NewsGuard, Handles the difference between news and opinion responsibly",
        "NewsGuard, Avoids deceptive headlines",
        "NewsGuard, Website discloses ownership and financing",
        "NewsGuard, Clearly labels advertising",
        "NewsGuard, Reveals who's in charge, including any possible conflicts of interest",
        "NewsGuard, Provides information about content creators",
        "NewsGuard, score",
        "NewsGuard, overall_class",
    ],
    "Pew Research Center": [
        "Pew Research Center, known_by_40%",
        "Pew Research Center, total",
        "Pew Research Center, consistently_liberal",
        "Pew Research Center, mostly_liberal",
        "Pew Research Center, mixed",
        "Pew Research Center, mostly conservative",
        "Pew Research Center, consistently conservative",
    ],
    "Wikipedia": ["Wikipedia, is_fake"],
    "Open Sources": [
        "Open Sources, reliable",
        "Open Sources, fake",
        "Open Sources, unreliable",
        "Open Sources, bias",
        "Open Sources, conspiracy",
        "Open Sources, hate",
        "Open Sources, junksci",
        "Open Sources, rumor",
        "Open Sources, blog",
        "Open Sources, clickbait",
        "Open Sources, political",
        "Open Sources, satire",
        "Open Sources, state",
    ],
    "Media Bias / Fact Check": [
        "Media Bias / Fact Check, label",
        "Media Bias / Fact Check, factual_reporting",
        "Media Bias / Fact Check, extreme_left",
        "Media Bias / Fact Check, right",
        "Media Bias / Fact Check, extreme_right",
        "Media Bias / Fact Check, propaganda",
        "Media Bias / Fact Check, fake_news",
        "Media Bias / Fact Check, some_fake_news",
        "Media Bias / Fact Check, failed_fact_checks",
        "Media Bias / Fact Check, conspiracy",
        "Media Bias / Fact Check, pseudoscience",
        "Media Bias / Fact Check, hate_group",
        "Media Bias / Fact Check, anti_islam",
        "Media Bias / Fact Check, nationalism",
    ],
    "Allsides": [
        "Allsides, bias_rating",
        "Allsides, community_agree",
        "Allsides, community_disagree",
        "Allsides, community_label",
    ],
    "BuzzFeed": ["BuzzFeed, leaning"],
    "Politifact": [
        "PolitiFact, Pants on Fire!",
        "PolitiFact, False",
        "PolitiFact, Mostly False",
        "PolitiFact, Half-True",
        "PolitiFact, Mostly True",
        "PolitiFact, True",
    ],
}


def nela_load_labels():
    labels_df = pd.read_csv(os.path.join(DATA_RAW_PATH, "nela", "labels.csv"))
    labels_df = labels_df.rename(columns={"Unnamed: 0": "Source"})
    return labels_df


def nela_labels_gtsource(labels_df, gt_source):
    """ only return label columns from specified ground truth source """
    return labels_df[GT_COLS[gt_source]]


def nela_load_articles():
    pass
