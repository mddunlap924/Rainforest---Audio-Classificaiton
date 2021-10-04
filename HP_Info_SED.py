from HPTuning.hp_tuning_inputs_sed import get_inputs
import os


class HP_Inputs:
    def __init__(self, hp_inputs, iteration_number):
        self.dataset = hp_inputs['dataset']
        self.model = hp_inputs['model']
        self.sed = hp_inputs['sed']
        self.base_trainable = hp_inputs['base_trainable']
        self.augment = hp_inputs['augment']
        self.fp = hp_inputs['fp']
        self.drop_connect_rate = hp_inputs['drop_connect_rate']
        self.lr_min = hp_inputs['learning_rate_min']
        self.lr_max = hp_inputs['learning_rate_max']
        self.fc_layer = hp_inputs['fc_layer']
        self.epochs = hp_inputs['epochs']
        self.lr_schedule = hp_inputs['lr_schedule']
        self.folds = hp_inputs['folds']
        self.batch_size = hp_inputs['batch_size']
        self.training = hp_inputs['training']
        self.mixup = hp_inputs['mixup']
        self.loss_weights = hp_inputs['loss_weights']
        self.bias_initializer = hp_inputs['bias_initializer']
        self.iteration_number = iteration_number


        # Save Path
        base_path = f'C:\Kaggle\RainForest_R0\HPTuning'
        if self.sed:
         model_path = os.path.join(os.path.join(base_path, self.dataset), self.model + '_SED')
        else:
            model_path = os.path.join(os.path.join(base_path, self.dataset), self.model)
        model_iteration_name = self.model + '_' + str(self.iteration_number)
        save_path = os.path.join(model_path, model_iteration_name)
        self.save_path = save_path

        # Height and Width of Spectrogram
        self.height = int(self.dataset.split('_')[0])
        self.width = int(self.dataset.split('_')[1])


# function to get unique values
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list


# Get combinations of HP Inputs and Return a List of Classes
def hp_combinations_to_run():
    hp_combinations = get_inputs()

    # Determine how my model iterations will be needed based on hp_combinations
    dataset_model = []
    for hp_combination in hp_combinations.items():
        dataset_model.append(hp_combination[1]['dataset'] + ' ' + hp_combination[1]['model'])

    unique_models = unique(dataset_model)

    model_counts = {}
    for unique_model in unique_models:
        model_counts[unique_model] = {'dataset': unique_model.split(' ')[0],
                                      'model': unique_model.split(' ')[1],
                                      'number': dataset_model.count(unique_model)
                                      }

    # Loop each unique model and get a class for it
    model_classes = []
    for model_count in model_counts.items():
        dataset = model_count[1]['dataset']
        model = model_count[1]['model']

        count = 0
        for hp_combination in hp_combinations.items():
            hp_dataset = hp_combination[1]['dataset']
            hp_model = hp_combination[1]['model']

            if (dataset == hp_dataset) and (model == hp_model):
                model_class = HP_Inputs(hp_combination[1], count)
                base_trainable = model_class.base_trainable
                fc_layer = model_class.fc_layer
                if base_trainable is False and fc_layer is None:
                    print('Base is NOT trainable AND no FC_Layer ---> NOT Adding to Combination of Models')
                else:
                    model_classes.append(model_class)
                count += 1

    return model_classes
