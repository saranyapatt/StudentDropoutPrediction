import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score, f1_score, fbeta_score
import pandas as pd

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=8)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size=14)

    plt.tight_layout()
    plt.ylabel('True label (Actual)')
    plt.xlabel('Predicted label')
    plt.show()

def divisive_clustering(data, max_clusters=3):
    indices = [np.arange(len(data))]
    while len(indices) < max_clusters:
        largest_idx = max(indices, key=lambda idx: len(idx))
        indices = [idx for idx in indices if not np.array_equal(idx, largest_idx)]
        subset = data[largest_idx]
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(subset)
        indices.append(largest_idx[km.labels_ == 0])
        indices.append(largest_idx[km.labels_ == 1])
    labels = np.zeros(len(data), dtype=int)
    for cluster_id, idx in enumerate(indices):
        labels[idx] = cluster_id
    return labels

def evaluate_clustering(labels, name, y_true):
    print(f"\n{'='*65}\n{name}\n{'='*65}")

    df_eval = pd.DataFrame({'Cluster': labels, 'Actual': y_true.values})
    cross_tab = pd.crosstab(df_eval['Cluster'], df_eval['Actual'])
    cross_tab.columns = ['Stay (0)', 'Dropout (1)']
    cross_tab['Total']        = cross_tab.sum(axis=1)
    cross_tab['Dropout_Rate'] = (cross_tab['Dropout (1)'] / cross_tab['Total'] * 100).round(1)
    print("\nCluster Purity:")
    print(cross_tab.to_string())

    # --- AUTO-TUNE: find threshold that maximizes F-beta(0.5) ---
    # F-beta with beta=0.5 weights PRECISION 2× more than recall
    dropout_rates = df_eval.groupby('Cluster')['Actual'].mean()
    overall_rate  = y_true.mean()

    # Try each cluster's dropout rate as a candidate threshold
    candidates = sorted(set(dropout_rates.values), reverse=True)

    best_fbeta = -1
    best_threshold = None
    best_mapping = None
    best_metrics = None

    for threshold in candidates:
        mapping = {cid: (1 if dropout_rates[cid] >= threshold else 0)
                   for cid in dropout_rates.index}
        if sum(mapping.values()) == 0:
            continue
        y_pred = df_eval['Cluster'].map(mapping).values

        prec   = precision_score(y_true.values, y_pred, zero_division=0)
        rec    = recall_score(y_true.values, y_pred, zero_division=0)
        f1     = f1_score(y_true.values, y_pred, zero_division=0)
        fbeta  = fbeta_score(y_true.values, y_pred, beta=0.5, zero_division=0)

        if fbeta > best_fbeta:
            best_fbeta     = fbeta
            best_threshold = threshold
            best_mapping   = mapping
            best_metrics   = (prec, rec, f1, fbeta)

    print(f"\nOverall dropout rate          : {overall_rate * 100:.1f}%")
    print(f"Auto-selected threshold       : {best_threshold * 100:.1f}%  "
          f"({best_threshold / overall_rate:.2f}× overall)")

    print("\nCluster → Label Mapping:")
    for cid, lbl in best_mapping.items():
        label_name = "Dropout ← at-risk" if lbl == 1 else "Stay"
        print(f"  Cluster {cid} → {label_name}  (rate: {dropout_rates[cid] * 100:.1f}%)")

    # --- Final metrics with best threshold ---
    y_pred = df_eval['Cluster'].map(best_mapping).values
    cm     = confusion_matrix(y_true.values, y_pred)
    acc    = accuracy_score(y_true.values, y_pred)
    prec, rec, f1, fbeta = best_metrics

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Stay  Predicted Dropout")
    print(f"  Actual Stay         {cm[0,0]:>6}            {cm[0,1]:>6}")
    print(f"  Actual Dropout      {cm[1,0]:>6}            {cm[1,1]:>6}")

    print(f"\nAccuracy   : {acc * 100:.2f}%")
    print(f"Precision  : {prec:.4f}   ← higher = fewer false alarms")
    print(f"Recall     : {rec:.4f}   ← higher = more dropouts caught")
    print(f"F1 Score   : {f1:.4f}")
    print(f"F-beta(0.5): {fbeta:.4f}  ← optimization target (precision-weighted)")

    plot_confusion_matrix(cm, classes=['Stay (0)', 'Dropout (1)'],
                          title=f'{name} — Confusion Matrix')

    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'fbeta': fbeta}
