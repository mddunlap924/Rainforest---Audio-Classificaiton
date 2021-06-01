import matplotlib.pyplot as plt
import os


# Plot Results
def plot_metrics(history):
    metrics = ['loss', 'lwlrap', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color='r', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color='b', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.5, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()


def plot_few_metrics(history):
    metrics = ['loss', 'lwlrap', 'bc_loss']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(1, 3, n + 1)
        plt.plot(history.epoch, history.history[metric], color='r', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color='b', linestyle="--", label='Val')
        if n == 1:
            plt.axhline(0.8, color='green')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.5, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()


# Normalize data to specified range
def normalize_image(x, a, b):
    x_min = x.min()
    x_max = x.max()
    x_norm = a + (((x - x_min) * (b - a)) / (x_max - x_min))
    return x_norm

# Plot SED model results
def plot_sed_few_metrics(history, save_path, model_name, model_result):
    metrics = ['loss', 'lwlrap']
    fig = plt.figure(figsize=(20, 10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        # plt.subplot(1, 3, n + 1)
        if metric == 'loss':
            plt.subplot(1, 3, 1)
            plt.plot(history.epoch, history.history[metric], color='r', label='Train')
            plt.plot(history.epoch, history.history['val_' + metric], color='b', label='Val')
            plt.title(f'Min Total: {model_result[metric]}')
            plt.ylabel('Total Loss')
            # plt.ylim([0, 0.09])
            plt.ylim([0, 0.14])

            plt.subplot(1, 3, 2)
            for metric_additional in ['clip', 'frame']:
                metric_to_plot = metric_additional + '_' + metric
                if 'clip' in metric_additional:
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
                plt.plot(history.epoch, history.history[metric_to_plot], color='r', label='Train', linestyle=linestyle)
                plt.plot(history.epoch, history.history['val_' + metric_to_plot], color='b', label='Val', linestyle=linestyle)
                plt.title(f'{model_result["clip_" + metric]}')
                plt.ylabel('Loss')
        if metric == 'f1':
            plt.subplot(1, 4, 3)
            for metric_additional in ['clip', 'frame']:
                metric_to_plot = metric_additional + '_' + metric_additional + '_' + metric
                if metric_additional == 'clip':
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
                plt.plot(history.epoch, history.history[metric_to_plot], color='r', linestyle=linestyle,
                         label='Train ' + metric_additional)
                plt.plot(history.epoch, history.history['val_' + metric_to_plot],
                         color='b', linestyle=linestyle, label='Val ' + metric_additional)
            plt.axhline(model_result['clip_' + metric], color='black')
            plt.title(f'Clip: {model_result["clip_" + metric]}')
            plt.ylabel(metric)
        if metric == 'lwlrap':
            plt.subplot(1, 3, 3)
            for metric_additional in ['clip', 'frame']:
                metric_to_plot = metric_additional + '_' + metric
                if metric_additional == 'clip':
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
                plt.plot(history.epoch, history.history[metric_to_plot], color='r', linestyle=linestyle,
                         label='Train ' + metric_additional)
                plt.plot(history.epoch, history.history['val_' + metric_to_plot],
                         color='b', linestyle=linestyle, label='Val ' + metric_additional)
            plt.axhline(model_result['clip_' + metric], color='black')
            plt.title(f'Clip: {model_result["clip_" + metric]}')
            plt.ylabel(metric)

        if metric == 'lwlrap':
            plt.axhline(0.85, color='green')

        plt.xlabel('Epoch')

        if metric == 'loss':
            # plt.ylim([0, 0.15])
            plt.ylim([0, 0.14])
        else:
            plt.ylim([0.75, 1])
        plt.legend()
    fig.suptitle(f'Frame LWLRAP {model_result["frame_lwlrap"]} \n'
                 f'Clip LWLRAP {model_result["clip_lwlrap"]}')
    plt.savefig(os.path.join(save_path, model_name + '.png'))

def plot_no_sed_few_metrics(history, save_path, model_name, model_result):
    metrics = ['loss', 'lwlrap', 'f1']
    fig = plt.figure(figsize=(20, 10))
    for n, metric in enumerate(metrics):
        if metric == 'loss':
            plt.subplot(1, 3, 1)
            plt.plot(history.epoch, history.history[metric], color='r', label='Train')
            plt.plot(history.epoch, history.history['val_' + metric], color='b', label='Val')
            plt.title(f'Min Total: {model_result[metric]}')
            plt.ylabel('Total Loss')
        if metric == 'lwlrap':
            plt.subplot(1, 3, 2)
            linestyle = 'solid'
            plt.plot(history.epoch, history.history[metric], color='r', label='Train', linestyle=linestyle)
            plt.plot(history.epoch, history.history['val_' + metric], color='b', label='Val', linestyle=linestyle)
            plt.axhline(model_result[metric], color='black')
            plt.axhline(0.85, color='green')
            plt.title(f'{model_result[metric]}')
            plt.ylabel('LWLRAP')
        if metric == 'f1':
            plt.subplot(1, 3, 3)
            linestyle = 'solid'
            plt.plot(history.epoch, history.history[metric], color='r', label='Train', linestyle=linestyle)
            plt.plot(history.epoch, history.history['val_' + metric], color='b', label='Val', linestyle=linestyle)
            plt.axhline(model_result[metric], color='black')
            plt.title(f'{model_result[metric]}')
            plt.ylabel('F1')
        plt.xlabel('Epoch')
        if metric == 'loss':
            plt.ylim([0, 0.2])
        else:
            plt.ylim([0, 1])
        plt.legend()
    fig.suptitle(f'LWLRAP {model_result["lwlrap"]}')
    plt.savefig(os.path.join(save_path, model_name + '.png'))
