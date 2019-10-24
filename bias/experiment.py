import datasets
import util

import lstm

def experiment_dataset(
    selection_problem,
    selection_source,
    selection_count,
    selection_random_seed,
    selection_overwrite,
    embedding_type,
    embedding_shape,
    embedding_overwrite
):
    # get selection set
    selection_df, name = datasets.get_selection_set(
        problem=selection_problem,
        source=selection_source,
        count=selection_count,
        random_seed=selection_random_seed,
        overwrite=selection_overwrite,
    )

    # create necessary embedding/vector form
    embedding_df = datasets.get_embedding_set(
        selection_df,
        embedding_type=embedding_type,
        output_name=name,
        shaping=embedding_shape,
        overwrite=embedding_overwrite,
    )

    return embedding_df

def experiment_model(df, model_type):
    lstm.create_model() 


if __name__ == "__main__":
    util.init_logging()
    experiment("reliability", "os", "10000", 13, False)
