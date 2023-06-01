import splitfolders


def split_dataset(input_path_to_data_for_split:str=None,
                 output_path_to_data_for_split:str=None,
                 my_ratio:tuple=(0.65,0.2,0.15),
                 my_seed:int=2):
    """_summary_

    Args:
        input_path_to_data_for_split (str, optional): _description_. Defaults to None.
        output_path_to_data_for_split (str, optional): _description_. Defaults to None.
        my_ratio (tuple, optional): _description_. Defaults to (0.65,0.2,0.15).
        my_seed (int, optional): _description_. Defaults to 2.
    """
    splitfolders.ratio(input_path_to_data_for_split,
                               output_path_to_data_for_split,
                               ratio=my_ratio,
                               seed=my_seed)