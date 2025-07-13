import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json

def generate_vibrant_colors(num_colors):
    """Generate a list of vibrant colors."""
    colors = plt.cm.get_cmap("hsv", num_colors)  # Use HSV colormap for vibrant colors
    return [colors(i)[:3] for i in range(num_colors)]  # Return RGB values

def compute_and_generate_plots(model, model_name, disease_name, train_time, history, test_generator, y_test, label_map):
    """
    Computes predictions, metrics, and generates/saves all plots and report.
    """

    base_dir = f'/content/drive/MyDrive/plots/{model_name}/{disease_name}'
    os.makedirs(base_dir, exist_ok=True)

    print("\033[92m[INFO::]\033[0m\n🔍 Evaluating model...")
    y_pred_probs = model.predict(test_generator)
    y_true = y_test
    target_names = list(label_map.keys())
    num_classes = len(target_names)

    if num_classes == 2:
        if y_pred_probs.ndim == 1 or y_pred_probs.shape[1] == 1:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            y_pred_probs = np.column_stack([1 - y_pred_probs, y_pred_probs])
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    if y_pred_probs.shape[1] != num_classes:
        raise ValueError(f"Model output shape {y_pred_probs.shape[1]} does not match classes {num_classes}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    sensitivity = recall

    cm = confusion_matrix(y_true, y_pred)
    specificities, fnrs, npvs = [], [], []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        fnr = FN / (TP + FN) if (TP + FN) != 0 else 0
        npv = TN / (TN + FN) if (TN + FN) != 0 else 0
        specificities.append(specificity)
        fnrs.append(fnr)
        npvs.append(npv)
    specificity = np.mean(specificities)
    fnr = np.mean(fnrs)
    npv = np.mean(npvs)

    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    auc_scores = []
    fpr_dict = {}
    tpr_dict = {}
    if num_classes == 2:
        print(y_true_bin.shape)
        for i in range(2):
            one_hot = np.eye(2)[y_true]
            fpr, tpr, _ = roc_curve(one_hot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
        mean_auc = np.mean(auc_scores)
    else:
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
        mean_auc = np.mean(auc_scores)

    # 1. Training History
    try:
      plt.figure(figsize=(14, 5))
      plt.subplot(1, 2, 1)
      plt.plot(history['accuracy'], color='#4e79a7', label='Train Accuracy')
      plt.plot(history['val_accuracy'], color='#f28e2b', label='Validation Accuracy')
      plt.title('Accuracy History', fontsize=14)
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(history['loss'], color='#e15759', label='Train Loss')
      plt.plot(history['val_loss'], color='#76b7b2', label='Validation Loss')
      plt.title('Loss History', fontsize=14)
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.tight_layout()
      plt.savefig(os.path.join(base_dir, 'training_history.png'))
      plt.show()
    except:
      print("\033[92m[INFO::]\033[0m\n\n[INFO]:: Can't plot the accuracy and loss history")
    # 2. Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
    plt.show()

    # 3. ROC Curves
    plt.figure(figsize=(8, 6))
    for i in range(len(fpr_dict)):
        plt.plot(fpr_dict[i], tpr_dict[i], label=f'{target_names[i]} (AUC={auc_scores[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title(f'ROC Curves ({num_classes}-Class)', fontsize=16)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'roc_curves.png'))
    plt.show()

    # 4. Metrics Summary
    metrics_list = [
        ('Accuracy', accuracy),
        ('Precision', precision),
        ('Recall/Sensitivity', recall),
        ('F1-Score', f1),
        ('Specificity', specificity),
        ('False Negative Rate', fnr),
        ('Negative Predictive Value', npv),
        ('AUC', mean_auc),
        ('Training Time', train_time)
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis('off')
    table_data = [[m[0], f'{m[1]:.4f}'] for m in metrics_list]
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     cellLoc='center', loc='center',
                     colColours=['#4e79a7', '#4e79a7'],
                     colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title('Performance Metrics Summary', fontsize=16)
    plt.savefig(os.path.join(base_dir, 'metrics_summary.png'))
    plt.show()

    # 5. Classification Report
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\033[92m[INFO::]\033[0m\n📋 Detailed Classification Report:")
    print(report)
    with open(os.path.join(base_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'fnr': fnr,
        'npv': npv,
        'auc': mean_auc,
        'training_time': train_time,
        'status': 'computed_new'
    }

def ensure_training_history_plot(model_name, disease_name):
    """
    Ensures the training history plot exists. If not, loads history from JSON and creates the plot.
    """
    base_dir = f'/content/drive/MyDrive/plots/{model_name}/{disease_name}'
    os.makedirs(base_dir, exist_ok=True)
    history_plot_path = os.path.join(base_dir, 'training_history.png')
    history_json_path = f'/content/drive/MyDrive/weights/{model_name}/{disease_name}/history.json'

    # Check if plot exists
    if os.path.exists(history_plot_path):
        print(f"\033[92m[INFO::]\033[0m✅ Training history plot already exists: {history_plot_path}")
        return

    # If not, try to load history and plot
    if os.path.exists(history_json_path):
        print(f"\033[92m[INFO::]\033[0m📂 Loading training history from {history_json_path}")
        with open(history_json_path, 'r') as f:
            history = json.load(f)
        # Plot
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], color='#4e79a7', label='Train Accuracy')
        plt.plot(history['val_accuracy'], color='#f28e2b', label='Validation Accuracy')
        plt.title('Accuracy History', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], color='#e15759', label='Train Loss')
        plt.plot(history['val_loss'], color='#76b7b2', label='Validation Loss')
        plt.title('Loss History', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(history_plot_path)
        plt.show()
        print(f"\033[92m[INFO::]\033[0m✅ Training history plot generated and saved: {history_plot_path}")
    else:
        print(f"\033[92m[INFO::]\033[0m❌ Neither plot nor history JSON found for {model_name}/{disease_name}.")

def classification_report_to_html(report_text):
    """
    Converts a scikit-learn classification_report string into a Bootstrap-styled HTML table.
    """
    import re
    lines = report_text.strip().split('\n')
    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Find lines with actual data (skip header lines)
    data_lines = []
    for line in lines:
        if re.match(r'^\s*(\w|\[)', line):  # line starts with a word or [
            if any(x in line for x in ['precision', 'recall', 'f1-score', 'support']):
                continue  # skip header
            if 'accuracy' in line:
                continue  # skip accuracy line (will handle separately)
            data_lines.append(line)

    # Extract accuracy line
    accuracy = None
    for line in lines:
        if line.strip().startswith('accuracy'):
            accuracy = re.findall(r'[\d\.]+', line)
            break

    # Build rows
    rows = []
    for line in data_lines:
        # Split by whitespace, but keep class names with spaces
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) == 5:
            rows.append(parts)

    # Add accuracy row if available
    if accuracy:
        rows.append(['accuracy', '', '', accuracy[0], accuracy[1]])

    # Table HTML
    table = [
        '<table class="table table-striped table-bordered">',
        '<thead><tr>',
        '<th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th>',
        '</tr></thead>',
        '<tbody>'
    ]
    for row in rows:
        table.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>')
    table.append('</tbody></table>')
    return '\n'.join(table)


def fetch_existing_plot_filepaths(model_name, disease_name):
    """
    Returns a dictionary of existing plot/report file paths for the given model and disease.
    Only includes files that actually exist.
    """
    base_dir = f'Datasets/plots/{model_name}/{disease_name}'
    base_dir2 = f'plots/{model_name}/{disease_name}'
    required_files = [
        'training_history.png',
        'confusion_matrix.png',
        'roc_curves.png',
        'metrics_summary.png',
        'classification_report.txt'
    ]
    existing_files = {}
    for file in required_files:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            file_path2 = os.path.join(base_dir2, file)
            existing_files[file.split('.')[0]]=file_path2
            
    return existing_files

def evaluate_and_visualize(model, model_name, disease_name, train_time=None, history=None, test_generator=None, y_test=None, label_map=None, fetch_existing=False):
    """
    Evaluates model and generates comprehensive visualizations and metrics.
    Skips computation if plots exist and fetch_existing=True.
    """
    ensure_training_history_plot(model_name,disease_name)
    if(fetch_existing):
      return fetch_existing_plot_filepaths(model_name,disease_name)
    else:
      return compute_and_generate_plots(model, model_name, disease_name, train_time, history, test_generator, y_test, label_map)
