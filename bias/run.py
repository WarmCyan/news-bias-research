import datasets
import util


def create_datasets():
    df_os_reliability, df_mbfc_reliability, df_ng_reliability = (
        datasets.get_selection_sets()
    )

    overwrite_w2v = False
    overwrite_glove = False
    overwrite_tfidf = True

    datasets.create_word2vec(df_os_reliability, "os_reliable_w2v", overwrite=overwrite_w2v)
    datasets.create_word2vec(df_mbfc_reliability, "mbfc_reliable_w2v", overwrite=overwrite_w2v)
    datasets.create_word2vec(df_ng_reliability, "ng_reliable_w2v", overwrite=overwrite_w2v)

    datasets.clear_vector_model()

    datasets.create_glove(df_os_reliability, "os_reliable_glove", overwrite=overwrite_glove)
    datasets.create_glove(df_mbfc_reliability, "mbfc_reliable_glove", overwrite=overwrite_glove)
    datasets.create_glove(df_ng_reliability, "ng_reliable_glove", overwrite=overwrite_glove)
    
    datasets.clear_vector_model()
    
    datasets.create_tfidf(df_os_reliability, "os_reliable_tfidf", overwrite=overwrite_tfidf)
    datasets.create_tfidf(df_mbfc_reliability, "mbfc_reliable_tfidf", overwrite=overwrite_tfidf)
    datasets.create_tfidf(df_ng_reliability, "ng_reliable_tfidf", overwrite=overwrite_tfidf)

    # datasets.create_word2vec(df_os_reliability, "os_reliable_glove_sequence", shaping="sequence")
    # datasets.create_word2vec(df_os_reliability, "os_reliable_glove_avg", shaping="avg")
    # 
    # datasets.create_word2vec(df_mbfc_reliability, "mbfc_reliable_glove_sequence", shaping="sequence")
    # datasets.create_word2vec(df_mbfc_reliability, "mbfc_reliable_glove_avg", shaping="avg")
    # 
    # datasets.create_word2vec(df_ng_reliability, "ng_reliable_glove_sequence", shaping="sequence")
    # datasets.create_word2vec(df_ng_reliability, "ng_reliable_glove_avg", shaping="avg")

if __name__ == "__main__":
    util.init_logging()
    create_datasets()
