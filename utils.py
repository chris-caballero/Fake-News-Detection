import evaluate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from datasets import load_dataset, Dataset
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, roc_curve

palette = sns.color_palette('hls')
truth_palette = sns.color_palette('hls', 15)[:6]

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
def concanenate_columns(dataset, max_len=100, splits=['train', 'test', 'validation'], cols=['statement', 'subject', 'speaker', 'context']):
    def truncate(sample, col, max_len):
        return {
            col: sample[col][:max_len] if len(sample[col]) > max_len else sample[col]
        }

    for split in splits:
        dataset[split] = dataset[split].map(lambda x: truncate(x, 'statement', max_len))
        columns = [dataset[split][col] for col in cols]
        concatenated_texts = [' [SEP] '.join(items) for items in zip(*columns)]
        dataset[split] = dataset[split].add_column('text', concatenated_texts)
    
    return dataset

# loads the dataset and applies the above functions and mappings
def load_data(path, split=None, remove_all=True, concat=False, max_len=75):
    if split is not None:
        dataset = load_dataset(path, split=split)
        dataset = dataset.map(map_labels)
        return dataset
    
    dataset = load_dataset(path)

    dataset['train'] = dataset['train'].map(map_labels)
    dataset['test'] = dataset['test'].map(map_labels)
    dataset['validation'] = dataset['validation'].map(map_labels)

    if concat:
        dataset = concanenate_columns(dataset, max_len=max_len)

    if remove_all:
        cols_to_remove = dataset['train'].column_names
        cols_to_remove.remove('label')
        cols_to_remove.remove('binary_label')
        cols_to_remove.remove('trinary_label')
        cols_to_remove.remove('label_name')
        cols_to_remove.remove('statement')
        if concat:
            cols_to_remove.remove('text')
        dataset = dataset.remove_columns(cols_to_remove)

    return dataset

# balances dataset by minority class
def balance_dataset(dataset, split='train'):
    # Separate features and labels
    if 'text' not in dataset[split].column_names:
        X = np.array(dataset[split]['statement'])
    else:
        X = np.array(dataset[split]['text'])
    y = np.array(dataset[split]['label'])

    # Reshape the features
    X_reshaped = X.reshape(-1, 1)

    # Apply undersampling to balance the classes based on the minority class
    sampler = RandomUnderSampler(sampling_strategy='auto')
    X_resampled, y_resampled, indices = sampler.fit_resample(X_reshaped, y)

    data_dict = {
        col: dataset[split][col][indices] for col in dataset[split].column_names
    }
    data_dict['text'] = X_resampled.squeeze(1)
    data_dict['label'] = y_resampled
    # Create a new balanced dataset
    balanced_dataset = Dataset.from_dict(data_dict)

    return balanced_dataset

def display_samples(dataset, num_samples=5):
    data = dataset.shuffle(seed=42).select(range(num_samples))

    for example in data:
        print(example['text'], example['label'])

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    print(labels)
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def evaluate_performance(trainer, inputs):
    from torch import from_numpy
    
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
    plt.show()

def get_roc_curve(trainer, inputs):
    y_pred = trainer.predict(inputs)
    y_proba = softmax(y_pred.predictions)

    y_true_labels = y_pred.label_ids

    fpr, tpr, threshold = roc_curve(y_true_labels, y_proba[:, 1])

    return fpr, tpr, threshold

def plot_roc_curves(model, inputs, label='Train Classifier'):
    fpr, tpr, _ = get_roc_curve(model, inputs)

    plt.figure(figsize=(7, 5))
    plt.title(f'ROC Curve for {label}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([-.05, 1, 0, 1.05])

    plt.plot(fpr, tpr, label=label, color=sns.color_palette('hls')[0])

    plt.legend()
    plt.show()