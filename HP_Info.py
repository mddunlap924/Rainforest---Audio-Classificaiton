from HPTuning.hp_tuning_inputs_sed import get_inputs
import os


class HP_Inputs:
    def __init__(self, hp_inputs, iteration_number):
        self.dataset = hp_inputs['dataset']
        self.model = hp_inputs['model']
        self.lr_min = hp_inputs['learning_rate_min']
        self.lr_max = hp_inputs['learning_rate_max']
        self.layer_combination = hp_inputs['layer_combination']
        self.regularization = hp_inputs['regularization']
        self.dropout0 = hp_inputs['dropout0']
        self.num_hidden = hp_inputs['num_hidden']
        self.dropout1 = hp_inputs['dropout1']
        self.patience = hp_inputs['patience']
        self.clr_step = hp_inputs['clr_step']
        self.output_bias = hp_inputs['output_bias']
        self.folds = hp_inputs['folds']
        self.batch_size = hp_inputs['batch_size']
        self.training = hp_inputs['training']
        self.clr_mode = hp_inputs['clr_mode']
        self.mixup = hp_inputs['mixup']
        self.iteration_number = iteration_number


        # Save Path
        base_path = f'C:\Kaggle\RainForest_R0\HPTuning'
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
                model_classes.append(model_class)
                count += 1

    return model_classes

# print('End of Script')

# inputs = hp_inputs[0]
# save_directory = os.path.join(os.path.join(os.path.join(os.getcwd(), 'HPTuning'), inputs['dataset']), inputs['model'])

# def path_exist(path):
#     if not os.path.isfile(model_path):
#         print(f'Path Does NOT Exist and Will be Created: {path}')
#     else:
#         print(f'Path DOES Exist: {path}')
#         return True
#
# # check if dataset directory exists
# directory_path = os.path.join(os.path.join(os.getcwd(), 'HPTuning'), inputs['dataset'])
# _ = path_exist(directory_path)
#
# # check if model directory is in dataset directory
# model_path = os.path.join(directory_path, inputs['model'])
# _ = path_exist(model_path)
#
# # check if model iteration exists in model directory - if does stop execution and move files to another location for
# # storate
# model_iteration_path = os.path.join(directory_path, inputs['model'])
# if path_exist(model_iteration_path):
#     print(f'Model Iteration Exists and Needs to Be Moved: {model_iteration_path}')
#     exit()
