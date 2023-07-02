from torch import save


def save_weights_nn(model_state_dict: dict,
                    name_model: str = None,
                    settings_network: str = None) -> None:
    if (name_model and settings_network) is None:
        raise Exception('Имя модели или настройки не заданы')
    else:
        save(model_state_dict, '/home/rain/vs_code/relize/saves_weights/' + name_model + '/' + name_model + '_' + settings_network + '.pth')


def save_all_model_nn(model,
                      name_model: str = None,
                      settings_network: str = None) -> None:
    if (name_model and settings_network) is None:
        raise Exception('Имя модели или настройки не заданы')
    else:
        save(model, '/home/rain/vs_code/relize/saved_trained_models/' + name_model + '/' + name_model + '_' + settings_network + '.pth')
