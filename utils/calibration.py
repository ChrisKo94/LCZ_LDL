### Adapted from https://github.com/hollance/reliability-diagrams ###

import numpy as np

def compute_calibration(true_labels, pred_labels, confidences, confidences_mat, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    bin_class_counts = np.zeros((num_bins, 10), dtype=np.int)
    bin_class_accuracies = np.zeros((num_bins, 10), dtype=np.float)
    bin_class_confidences = np.zeros((num_bins, 10), dtype=np.float)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
            for k in range(10):
                bin_class_counts[b, k] = np.sum(true_labels[selected] == k + 1)
                if bin_class_counts[b, k] > 0:
                    bin_class_accuracies[b, k] = np.mean((true_labels[selected] == pred_labels[selected])[true_labels[selected] == k + 1])
                    bin_class_confidences[b, k] = np.mean(confidences_mat[selected, k][true_labels[selected] == k + 1])

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)

    class_gaps = np.abs(bin_class_accuracies - bin_class_confidences)

    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    sce = 1/10 * np.sum(class_gaps * bin_class_counts) / np.sum(bin_class_counts)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce,
            "static_calibration_error": sce}
