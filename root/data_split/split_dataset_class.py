from typing import Any

import splitfolders


class My_split_folders:
    def __init__(self, input_path_to_data_for_split:str=None,
                 output_path_to_data_for_split:str=None,
                 my_ratio:tuple=(0.65,0.2,0.15),
                 my_seed:int=2,
                 my_group_prefix:Any | None=None,
                 my_move:bool=False
                 )->str:
        if input_path_to_data_for_split or output_path_to_data_for_split is None:
            raise FileNotFoundError("Пути к папкам не заданы")
        try:
            splitfolders.ratio(input_path_to_data_for_split,
                               output_path_to_data_for_split,
                               ratio=my_ratio,
                               seed=my_seed,
                               group_prefix=my_group_prefix,
                               move=my_move)
            print(f'Выходная папка: {output_path_to_data_for_split}')
        except BaseException as e:
            print(f'Non standart situation {e}')
