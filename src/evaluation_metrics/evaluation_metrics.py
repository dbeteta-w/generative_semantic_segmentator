import numpy as np


def compute_intersection_over_union(annotation, mask):
    """ IoU: Intersection (AND) / Union (OR) """

    # Compute intersection
    intersection = np.sum(np.logical_and(annotation, mask))

    # Compute union
    union = np.sum(np.logical_or(annotation, mask))

    # Compute intersection over union
    iou = intersection / union

    return iou


def compute_balanced_accuracy(annotation, mask):
    """ BA: 2*Intersection - Union (TP-FP-FN) / Ground Truth (Annotation) """

    # Compute intersection
    intersection = np.sum(np.logical_and(annotation, mask))

    # Compute union
    union = np.sum(np.logical_or(annotation, mask))

    # Compute pixel accuracy
    pa_score = (2*intersection - union) / np.sum(annotation)