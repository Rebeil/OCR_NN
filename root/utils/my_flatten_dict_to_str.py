#from utils.load_config_yaml import load_settings
#from load_config_yaml import load_settings


def list_to_flatten_str(nested_list: list) -> str:
    return '_'.join(str(x) for x in nested_list)


def my_recursive(obj: dict, list_items=None) -> list:
    if list_items is None:
        list_items = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            list_items.append(key)
            my_recursive(value)
    else:
        list_items.append(obj)
    return list_items


def recursive_dict_to_str(obj: dict, list_items: list = []) -> str:
    # list_items = list_items if list_items and isinstance(list_items, list) else []
    # if list_items is not None and isinstance(list_items, list):
    #     list_items += list_items
    # else:
    #     list_items = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            list_items.append(key)
            recursive_dict_to_str(value)
    else:
        list_items.append(obj)
    # print(list_items)
    return '_'.join(str(x) for x in list_items)

#
# if __name__ == '__main__':
#     settings = load_settings()
#     # e = my_recursive(settings['network_settings'])
#     # print(list_to_flatten_str(e))
#     # print(recursive_dict_to_str(settings['network_settings']))
#     # print(list_to_flatten_str(my_recursive(settings['network_settings'])))
#     # print(my_recursive(settings['network_settings']))
#     print(recursive_dict_to_str(settings['nn_set']))
