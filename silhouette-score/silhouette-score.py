import numpy as np

def silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    dists = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

    s_i_scores = []

    for i in range(n_samples):
        label_i = labels[i]

        same_cluster = (labels == label_i)
        same_cluster[i] = False  
        
        if np.any(same_cluster):
            a_i = np.mean(dists[i, same_cluster])
        else:
            a_i = 0

        b_i = min(
            np.mean(dists[i, labels == other_label])
            for other_label in unique_labels if other_label != label_i
        )

        s_i_scores.append((b_i - a_i) / max(a_i, b_i))

    return np.mean(s_i_scores)