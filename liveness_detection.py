# Full single-file code for Keystroke Liveness Detection using ARFF files
import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Data Loading and Preprocessing

def load_arff_dataset(base_path):
    all_data = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.arff'):
                filepath = os.path.join(root, file)
                data, meta = arff.loadarff(filepath)
                df = pd.DataFrame(data)
                # Find the label column dynamically
                label_col = None
                for col in df.columns:
                    if df[col].dtype == object:
                        label_col = col
                        break
                if label_col is None:
                    raise ValueError(f"No label column found in {file}")
                df[label_col] = df[label_col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                df.rename(columns={label_col: 'label'}, inplace=True)
                all_data.append(df)
    if not all_data:
        raise ValueError(f"No valid ARFF files found in {base_path}")
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Model Training and Evaluation

def train_liveness_detector(X, y):
    model = make_pipeline(StandardScaler(), SVC(kernel='poly', probability=True))
    model.fit(X, y)
    return model


def evaluate_detector(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, preds))

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Calculate EER (Equal Error Rate)
    eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")

# Main Execution

if __name__ == "__main__":
    base_path = './data'  # Change to your base dataset path

    print("Loading ARFF datasets...")
    df = load_arff_dataset(base_path)

    print("Preparing data...")
    X = df.drop(columns=['label']).values
    y = df['label'].apply(lambda x: 1 if x == 'real' else 0).values

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training liveness detector...")
    model = train_liveness_detector(X_train, y_train)

    print("Evaluating model...")
    evaluate_detector(model, X_test, y_test)