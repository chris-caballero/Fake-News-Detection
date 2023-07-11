import evaluate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import tensor, from_numpy
from torch.nn.functional import softmax
from datasets import load_dataset, Dataset
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, roc_curve

palette = sns.color_palette('hls')
truth_palette = sns.color_palette('hls', 15)[:6]

multiclass = {
    'binary_label': 2,
    'trinary_label': 3,
    'default': 6, 
}

mappings = {
    'two-class': {
        'pants-on-fire': 1,
        'false': 1,
        'barely-true': 1,
        'half-true': 0,
        'mostly-true': 0,
        'true': 0
    },
    'three-class': {
        'pants-on-fire': 2,
        'false': 2,
        'barely-true': 1,
        'half-true': 1,
        'mostly-true': 0,
        'true': 0
    },
    'default': {
        'pants-on-fire': 5,
        'false': 0,
        'barely-true': 4,
        'half-true': 1,
        'mostly-true': 2,
        'true': 3
    }
}

mappings_inverse = {
    mappings['default'][k]: k for k in mappings['default']
}


# Function to map labels to binary or trinary labels (classes is a list)
# Default simply maps labels to label name (as used in LIAR dataset)
def map_labels(sample):
    label_name = mappings_inverse[sample['label']]
    sample['label_name'] = label_name
    sample['binary_label'] = mappings['two-class'][label_name]
    sample['trinary_label'] = mappings['three-class'][label_name]

    return sample

# Concatenate columns with [SEP] to add more information for classification
def concanenate_columns(dataset, max_len=100, cols=['statement', 'subject', 'speaker', 'context']):
    def truncate(sample, col, max_len):
        return {
            col: sample[col][:max_len] if len(sample[col]) > max_len else sample[col]
        }
    
    dataset = dataset.map(lambda x: truncate(x, 'statement', max_len))
    columns = [dataset[col] for col in cols]
    concatenated_texts = [' [SEP] '.join(items) for items in zip(*columns)]
    dataset = dataset.add_column('text', concatenated_texts)
    
    return dataset

def load_split(path=None, data=None, remove_all=True, max_len=100, split='train', label_type='label', cols=['statement', 'subject', 'speaker', 'context']):
    if data is not None:
        dataset = Dataset.from_pandas(data)
    else:
        dataset = load_dataset(path, split=split)
        
    dataset = dataset.map(map_labels)
    dataset = concanenate_columns(dataset, max_len=max_len, cols=cols)
    if remove_all:
        cols_to_remove = dataset.column_names
        cols_to_remove.remove(label_type)
        cols_to_remove.remove('text')
        dataset = dataset.remove_columns(cols_to_remove)
    
    if label_type != 'label':
        dataset = dataset.rename_column(label_type, 'label')
    
    return dataset
        
        

# loads the dataset and applies the above functions and mappings
def load_data(path=None, data=None, split=None, remove_all=True, max_len=100, label_type='label', cols=['statement', 'subject', 'speaker', 'context']):
    if split is not None:
        dataset = load_split(path=path, split=split, cols=cols, label_type=label_type)
    elif data is not None:
        dataset = load_split(data=data, cols=cols, label_type=label_type)
    else:
        dataset = {}
        for split in ['train', 'test', 'validation']:
            dataset[split] = load_split(path=path, split=split, cols=cols, label_type=label_type)
    
    return dataset

# balances dataset by minority class
def balance_dataset(dataset, split=None):
    if split is not None:
        data = dataset[split]
    else:
        data = dataset
        
    # Separate features and labels
    if 'text' not in data.column_names:
        X = np.array(data['statement'])
    else:
        X = np.array(data['text'])
    y = np.array(data['label'])

    # Reshape the features
    X_reshaped = X.reshape(-1, 1)

    # Apply undersampling to balance the classes based on the minority class
    sampler = RandomUnderSampler(sampling_strategy='auto')
    X_resampled, y_resampled = sampler.fit_resample(X_reshaped, y)

    # Create a new balanced dataset
    balanced_dataset = Dataset.from_dict({
        'text': X_resampled.squeeze(1),
        'label': y_resampled
    })


    return balanced_dataset

def display_samples(dataset, num_samples=5):
    data = dataset.shuffle(seed=42).select(range(num_samples))

    for example in data:
        print(example['text'], example['label'])

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def evaluate_performance(trainer, inputs, label='Train Classifier'):
    # basic evaluation of model
    trainer.evaluate(inputs)

    # get label probability prediction
    y_pred = trainer.predict(inputs)
    pred = from_numpy(y_pred.predictions)

    y_proba = softmax(pred)

    # get predicted labels and true labels
    y_true_labels = y_pred.label_ids
    y_pred_labels = np.argmax(y_proba, axis=1)

    for metric in [evaluate.load("f1"), evaluate.load("recall"), evaluate.load("precision")]:
        print(metric.compute(predictions=y_pred_labels, references=y_true_labels, average='macro'))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    sns.heatmap(cm, annot=True)
    plt.title(f'Confusion Matrix for {label}')
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.show()

def get_roc_curve(trainer, inputs):
    y_pred = trainer.predict(inputs)
    y_proba = softmax(tensor(y_pred.predictions), dim=1)

    y_true_labels = y_pred.label_ids

    fpr, tpr, threshold = roc_curve(y_true_labels, y_proba[:, 1])

    return fpr, tpr, threshold


def plot_roc_curves(trainer, inputs, label='Train Classifier'):
    def plot_roc_curve(fpr, tpr, label=None, color='b'):
        plt.plot(fpr, tpr, label=label, color=color)

    fpr, tpr, threshold = get_roc_curve(trainer, inputs)

    plt.figure(figsize=(7, 5))
    plt.title(f'ROC Curve for {label}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([-.05, 1, 0, 1.05])


    palette = sns.color_palette('hls', 8)
    plot_roc_curve(fpr, tpr, label=label, color=sns.color_palette('hls')[0])

    plt.legend()
    plt.show()

def display_counts(dataset, split=None):
    if split:
        sns.countplot(data=pd.DataFrame(dataset[split]), x='label', palette=palette)
    else:
        sns.countplot(data=pd.DataFrame(dataset), x='label', palette=palette)
        
    plt.title('Label Count')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.show()

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)