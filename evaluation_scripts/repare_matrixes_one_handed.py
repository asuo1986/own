#! /usr/bin/python

from scipy.spatial import distance
import statsmodels.api as sm
import time
import xml.etree.ElementTree
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
import os
import tempfile
import shutil
from copy import copy

def load_labels(path):
    """Loads text labels from path.

    Loads text labels from text path as list of lists with 3 labels per each
    list [left_hand_label, right_hand_label, both_hands_label].

    Args:
        path: where text labels will be loaded from.

    Returns:
        A list of lists with 3 text labels per each list.

        [['point', 'none', 'none'],
        ['no_hand', 'fist', 'none']]

    """

    labels = []
    with open(path, "r") as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split(" ")
            labels.append(line)
    return labels


def load_translations(path):
    """Loads translations from path.

    Loads text translations from text path as a dictionary.

    Args:
        path: where text labels will be loaded from.

    Returns:
        A dictionary that maps a text label to an integer value.

    """

    text_to_label = {}
    unique_labels = []
    with open(path + ".separate") as f:
        content = f.readlines()

        for i in range(len(content)):
            unique_labels.append(' ')

        for line in content:
            line = line.strip().split(" ")
            text_to_label[line[1]] = int(line[0])
            unique_labels[int(line[0])] = line[1]


    return {'ttl':text_to_label , 'ul': unique_labels}


def load_estimated_histograms(path):
    """Loads estimated histograms from path.

    Loads estimated histograms from text path as a list of lists.

    Args:
        path: where text labels will be loaded from.

    Returns:
        A list of lists. Each list contains concatenated histogram for left,
        right, both hands.

    """

    histograms = []
    with open(path, "r") as f:
        content = f.readlines()
        for line in content:
            histograms.append([float(x) for x in line.strip().split(" ")])
    return histograms


def get_hand_type_gesture(left, right):
    """Gets merged label that represents hand type.

    Gets merged label (hand type) as result of left and right hand gestures
    combination.

    Args:
        left: left hand gesture.
        right: right hand gesture.

    Returns:
        A string that represents hand type.

    """

    if left == "no_hand" and right == "no_hand":
        return "no_hand"
    if left == "no_hand" and right != "no_hand":
        return "right"
    if left != "no_hand" and right == "no_hand":
        return "left"
    if left != "no_hand" and right != "no_hand":
        return "both"


def plot_confusion_matrix(cm, classes_map, title="Confusion matrix", trash_column=[],
                          cmap=plt.cm.Blues):
    """
    Plots confusion matrix.

    :param cm: Confusion matrix.
    :param title: Figure title.
    :param cmap: Color map to be applied to confusion matrix values.
    :return:
    """
    cmap.set_bad("Khaki", 1.)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "black"
            if value > 50.0:
                color = "white"
            if not np.isnan(value):
                plt.text(j, i, "{:.1f}".format(value),
                        horizontalalignment="center",
                        verticalalignment="center", color=color)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes_map))
    plt.xticks(tick_marks, classes_map, rotation=90)
    plt.yticks(tick_marks, classes_map)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

def plot_hands_typr_error_col (error_col, title="Hands type mis", cmap=plt.cm.Blues):
    cmap.set_bad("Khaki", 1.)
    plt.imshow(error_col, interpolation="nearest", cmap=cmap)
    for i in range(error_col.shape[0]):
        for j in range(error_col.shape[1]):
            value = error_col[i, j]
            color = "black"
            if value > 50.0:
                color = "white"
            if not np.isnan(value):
                plt.text(j, i, "{:.1f}".format(value),
                        horizontalalignment="center",
                        verticalalignment="center", color=color)
    plt.title(title)
    plt.tight_layout()



def get_conf_mat_and_misclassifications(gt_labels, pr_labels, text_to_label,
                                        misclassifications):
    """Returns confusion matrix and list of misclassifications.

    Args:
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        text_to_label: a map from a text label to an integer value.
        misclassifications: list of misclassifications.

    Returns:
        Confusion matrix as 2D numpy array.

    """

    cm = np.zeros((len(text_to_label), len(text_to_label)))

    idx = 0
    for gt_label, pr_label in zip(gt_labels, pr_labels):
        cm[text_to_label[gt_label]][text_to_label[pr_label]] += 1
        if text_to_label[gt_label] != text_to_label[pr_label]:
            misclassifications.append(idx)
        idx += 1

    cm = np.array(cm)
    cm_normalized = 100.0 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized, 3)
    return cm_normalized

def get_conf_mat_and_misclassifications_non_norm(gt_labels, pr_labels, text_to_label,
                                        misclassifications):
    """Returns confusion matrix and list of misclassifications.

    Args:
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        text_to_label: a map from a text label to an integer value.
        misclassifications: list of misclassifications.

    Returns:
        Confusion matrix as 2D numpy array.

    """

    cm = np.zeros((len(text_to_label), len(text_to_label)))

    idx = 0
    for gt_label, pr_label in zip(gt_labels, pr_labels):
        cm[text_to_label[gt_label]][text_to_label[pr_label]] += 1
        if text_to_label[gt_label] != text_to_label[pr_label]:
            misclassifications.append(idx)
        idx += 1

    cm = np.array(cm)
    return cm



def plot_precision_recall_curves(gt_labels, text_labels, histograms, title):
    """Plots a precision-recall curve.

    Args:
        gt_labels: ground truth labels list.
        text_labels: a list that contains text labels.
        histograms: a list of predicted histograms.
        title: a title for plot.

    """
    unique_gt_labels = set(gt_labels)

    histograms = np.array(histograms)
    curves = []
    for i in unique_gt_labels:
        gt_binarized = [int(x == i) for x in gt_labels]
        precision, recall, _ = \
            precision_recall_curve(gt_binarized, histograms[:, i])

        area = average_precision_score(gt_binarized, histograms[:, i],
                                       average='micro')
        if not np.isnan(area):
            plt.plot(recall, precision, linewidth=1, label='{0} (AUC {1:0.3f})'
                     .format(text_labels[i], area))
            curves.append((recall[::-1], precision[::-1]))
    xvals = np.arange(0, 1, 0.01)
    avgvals = np.mean([interp1d(c[0], c[1])(xvals) for c in curves], axis=0)
    plt.plot(xvals, avgvals, label='mean (AUC {:.3})'.
             format(auc(xvals, avgvals)), color='gray', linewidth=2)

    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()


def plot_hand_type_conf_mat(path, gt_labels, pr_labels, misclassifications):
    """Plots a confusion matrix for hand type classification task.

    Args:
        path: where a confusion matrix will be saved to.
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        misclassifications: a list of misclassifications.

    """
    unique_text_labels = ["no_hand", "left", "right", "both"]
    text_to_label = {}

    for i in range(len(unique_text_labels)):
        text_to_label[unique_text_labels[i]] = i

    new_gt_labels = []
    for label in gt_labels:
        new_gt_labels.append(get_hand_type_gesture(label[0], label[1]))

    new_pr_labels = []
    for label in pr_labels:
        new_pr_labels.append(get_hand_type_gesture(label[0], label[1]))

    cm = get_conf_mat_and_misclassifications(new_gt_labels, new_pr_labels,
                                             text_to_label, misclassifications)

    plot_confusion_matrix(cm, unique_text_labels, "Hand type conf mat")
    plt.savefig(path + "_cm.png")
    plt.clf()


def plot_conf_mat_and_precision_recall(path, gt_labels, pr_labels,
                                       estimated_hists, text_to_label,
                                       misclassifications,
                                       hand_type_misclassifications=[]):
    """Plots a confusion matrix and precision-recall curves.

    PLots a confusion matrix and precision-recall curves for gesture
    classification task and returns list of misclassifications.

    Args:
        path: where plots will be saved to.
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        estimated_hists: estimated histograms list.
        misclassifications: a list of gesture misclassifications.
        hand_type_misclassifications: a list of hand type misclassifications.

    """

    sep = '/'

    # Get unique merged text labels.
    unique_text_labels = set()

    hand_type_right_classified = set()
    [hand_type_right_classified.add(i) for i in range(len(gt_labels))]
    hand_type_right_classified = \
        hand_type_right_classified.difference(hand_type_misclassifications)
    hand_type_right_classified = list(hand_type_right_classified)

    for i in hand_type_right_classified:
        label = gt_labels[i]
        unique_text_labels.add(label[0] + sep + label[1])

    for i in hand_type_right_classified:
        label = pr_labels[i]
        unique_text_labels.add(label[0] + sep + label[1])

    unique_text_labels = sorted(list(unique_text_labels))
    print 'initial unic labels', unique_text_labels


    # Get from text label to integer value mapping.
    new_text_to_label = {}
    for i in range(len(unique_text_labels)):
        new_text_to_label[unique_text_labels[i]] = i

    # Compute confusion matrix and save it to file.
    new_gt_labels = []
    new_pr_labels = []
    for i in hand_type_right_classified:
        new_gt_labels.append(gt_labels[i][0] + sep + gt_labels[i][1])
        new_pr_labels.append(pr_labels[i][0] + sep + pr_labels[i][1])

    # cm = get_conf_mat_and_misclassifications(new_gt_labels, new_pr_labels,
    #                                          new_text_to_label,
    #                                          misclassifications)

    cm = get_conf_mat_and_misclassifications_non_norm(new_gt_labels, new_pr_labels,
                                                      new_text_to_label,
                                                      misclassifications)

    # Prepare unique gestures
    unique_gestures = []
    for label in unique_text_labels:
        slash_pos = label.find('/')
        left_gesture = label[:slash_pos]
        right_gesture = label[slash_pos+1:]
        should_add_left = True
        should_add_right = True
        for gesture in unique_gestures:
            if gesture == left_gesture:
                should_add_left = False
            if gesture == right_gesture:
                should_add_right = False
        if should_add_left:
            unique_gestures.append(left_gesture)
        if should_add_right and (not right_gesture == left_gesture):
            unique_gestures.append(right_gesture)

    # remove non target combinations
    temp_cm = cm
    col_both_hand = np.zeros(len(unique_text_labels))
    temp_labels = unique_text_labels
    should_cut = True
    while should_cut:
        rows, cols = temp_cm.shape
        one_col_cuted = False
        col = 0
        while (not one_col_cuted) and (col < cols):
            label = temp_labels[col]
            slh = label.find('/')
            left = label[:slh]
            right = label[slh+1:]
            if not ((left == 'no_hand' and not right == 'no_hand') or (right == 'no_hand' and not left == 'no_hand')):
                for c in range(cols):
                    col_both_hand[c] += temp_cm[c][col]

                col_both_hand = np.delete(col_both_hand, col)
                temp_labels = np.delete(temp_labels, col)
                temp_cm = np.delete(temp_cm, col, 0)
                temp_cm = np.delete(temp_cm, np.s_[col], 1)

                one_col_cuted = True
            should_cut = one_col_cuted
            col += 1



    #merge correspondent labels
    # rows, cols = temp_cm.shape
    # temp_col_both_hands = col_both_hand
    # new_labels = temp_labels
    # col = 0
    # while col < cols:
    #     sh = new_labels[col].find('/')
    #     left = new_labels[col][:sh]
    #     right = new_labels[col][sh+1:]
    #
    #     gesture = ''
    #     if left == 'no_hand':
    #         gesture = right
    #     else:
    #         gesture = left
    #
    #     pair = -1
    #     for c in range(col + 1, cols):
    #         idx = new_labels[c].find(gesture)
    #         if  not idx == -1:
    #             pair = c
    #
    #     if not pair == -1:
    #         for r in range(rows):
    #             if not r == pair:
    #                 temp_col_both_hands[r] += temp_cm[r][pair]
    #         temp_cm[col][col] += temp_cm[pair][pair]
    #         temp_cm = np.delete(temp_cm, np.s_[pair], 1)
    #         rows, cols = temp_cm.shape
    #
    #         new_labels = np.delete(new_labels, pair)
    #         new_labels[col] = gesture
    #
    #         for c in range(cols):
    #             devid = new_labels[c].find('/')
    #             if devid != -1:
    #                 cur_left = new_labels[c][:devid]
    #                 cur_rite = new_labels[c][devid+1:]
    #
    #                 find_consistent_label = cur_rite + '/'+ cur_left
    #
    #                 for k in range(len(new_labels)):
    #                     if find_consistent_label == new_labels[k]:
    #                         temp_cm[col][k] += temp_cm[pair][c]
    #             else:
    #                 #  if this is avg column this should add
    #                 temp_cm[col][c] += temp_cm[pair][c]
    #         temp_cm = np.delete(temp_cm, pair, 0)
    #         rows, cols = temp_cm.shape
    #
    #         temp_col_both_hands[col] += temp_col_both_hands[pair]
    #         temp_col_both_hands = np.delete(temp_col_both_hands, pair)
    #     else:
    #         new_labels[col] = gesture
    #
    #     rows, cols = temp_cm.shape
    #     col += 1

    rows, cols = temp_cm.shape

    avg_matrix = np.zeros((rows, len(unique_gestures) + 1), dtype=np.float)
    rows_avg, cols_avg = avg_matrix.shape

    row_labels = []
    for i in range(len(temp_labels)):
        sh = temp_labels[i].find('/')
        left = temp_labels[i][:sh]
        right = temp_labels[i][sh+1:]
        if left != 'no_hand':
            gesture = left
        else:
            gesture = right
        row_labels.append(gesture)

    last_col = cols_avg - 1

    for i in range(rows):
        cur_label = temp_labels[i]
        sh = cur_label.find('/')
        left = cur_label[:sh]
        right = cur_label[sh+1:]
        gesture = ''
        rg = False
        if left == 'no_hand':
            gesture = right
            rg = True
        else:
            rg = False
            gesture = left

        label_pattern = ''
        if rg:
            label_pattern = 'no_hand/'
        else:
            label_pattern = '/no_hand'

        avg_matrix[i][last_col] += col_both_hand[i]
        for j in range(cols):
            ptr_pos = temp_labels[j].find(label_pattern)
            if ptr_pos == -1:
                avg_matrix[i][last_col] += temp_cm[i][j]
            else:
                temp_sh_pos = temp_labels[j].find('/')
                temp_left = temp_labels[j][:temp_sh_pos]
                temp_right = temp_labels[j][temp_sh_pos+1:]
                temp_gesture = ''
                if rg:
                    temp_gesture = temp_right
                else:
                    temp_gesture = temp_left

                for k in range(cols_avg - 1):
                    if unique_gestures[k] == temp_gesture:
                        avg_matrix[i][k] += temp_cm[i][j]

    # shrink rows
    rows, cols = avg_matrix.shape
    for i in range(rows):
        j = i+1
        should_find = True
        while should_find and (j < rows):
            if row_labels[i] == row_labels[j]:
                should_find = False
                for k in range(cols):
                    avg_matrix[i][k] += avg_matrix[j][k]
                avg_matrix = np.delete(avg_matrix, j, axis=0)
                row_labels = np.delete(row_labels, j)
                rows, cols = avg_matrix.shape
            j+=1
    nh_del_id = 0
    for i in range(len(unique_gestures)):
        if unique_gestures[i] == 'no_hand':
            nh_del_id = i
    unique_gestures = np.delete(unique_gestures, nh_del_id)
    avg_matrix = np.delete(avg_matrix, np.s_[nh_del_id], axis=1)
    rows, cols = avg_matrix.shape



    temp_col_both_hands = []
    for i in range(rows):
        temp_col_both_hands.append(avg_matrix[i][last_col - 1])

    avg_matrix = np.delete(avg_matrix, np.s_[last_col-1], axis=1)

    print 'Befor of the reordering:'
    print 'rows ', row_labels
    print 'cols ', unique_gestures

    ### reorder columns

    ## jnly for test
    reordered_col_labels = copy(unique_gestures)

    rows, cols = avg_matrix.shape
    reordered_matrix = np.zeros((rows, rows), dtype=np.float)
    for i in range(rows):
        row_gest = row_labels[i]
        for j in range(cols):
            if unique_gestures[j] == row_gest:
                reordered_col_labels[i] = unique_gestures[j];
                for k in range(rows):
                    reordered_matrix[k][i] = avg_matrix[k][j]
    avg_matrix = copy(reordered_matrix)

    print 'rows ', row_labels
    print 'cols ', reordered_col_labels

    temp_cm = copy(avg_matrix)
    new_labels = copy(reordered_col_labels)
    rows, cols = avg_matrix.shape

    # represent matrix as binary task gesture vs none
    none_ids = -1
    # for i in range(len(unique_gestures)):
    #     if unique_gestures[i] == 'none':
    #         none_ids = i
    # for i in range(len(unique_gestures)):
    #     for j in range(len(unique_gestures)):
    #         if j != none_ids and i != j:
    #             temp_cm[i][none_ids] += temp_cm[i][j]
    #             temp_cm[i][j] = 0


    temp_cm_norm = temp_cm
    temp_tresh_col = []
    for i in range(len(temp_col_both_hands)):
        temp_tresh_col.append([temp_col_both_hands[i]])

    for row in range(cols):
        sm = 0
        for col in range(cols):
            sm += temp_cm[row][col]
        sm += temp_col_both_hands[row]

        for col in range(cols):
            temp_cm_norm[row][col] = (100.0*temp_cm[row][col])/sm

        temp_tresh_col[row][0] = (100.0 * temp_tresh_col[row][0])/sm

    temp_cm_norm = np.round(temp_cm_norm, 3)
    temp_tresh_col = np.round(temp_tresh_col, 3)

    for i in range(len(misclassifications)):
        misclassifications[i] = \
            hand_type_right_classified[misclassifications[i]]

    plot_confusion_matrix(temp_cm_norm, new_labels, "Hand pose conf mat")
    plt.savefig(path + "_cm.png")
    plt.clf()

    plot_hands_typr_error_col (temp_tresh_col, title="Hands type mis")
    plt.savefig(path + "_cm_hmis.png")
    plt.clf()

    # Compute histograms for merged text labels.
    # right_hist_start_idx = len(text_to_label)
    # new_estimated_hists = []
    # for i in hand_type_right_classified:
    #     hist = estimated_hists[i]
    #     new_hist = []
    #     for label in unique_text_labels:
    #         label = label.split(sep)
    #         h = min(hist[text_to_label[label[0]]],
    #                 hist[right_hist_start_idx + text_to_label[label[1]]])
    #         new_hist.append(h)
    #     hist_sum = sum(new_hist)
    #     if hist_sum > 0:
    #         new_hist = [x / hist_sum for x in new_hist]
    #     new_estimated_hists.append(new_hist)

    new_gt_labels_int = []
    for label in new_gt_labels:
        new_gt_labels_int.append(new_text_to_label[label])

        #plot_precision_recall_curves(new_gt_labels_int, new_text_to_label.keys(),
        #new_estimated_hists, "Precision recall")
    plt.savefig(path + "_pr.png")
    plt.clf()

def plot_conf_mat_and_precision_recall_hand_pose(path, gt_labels, pr_labels,
                                       estimated_hists, text_to_label,
                                       misclassifications,
                                       hand_type_misclassifications=[]):
    """Plots a confusion matrix and precision-recall curves.

    PLots a confusion matrix and precision-recall curves for gesture
    classification task and returns list of misclassifications.

    Args:
        path: where plots will be saved to.
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        estimated_hists: estimated histograms list.
        misclassifications: a list of gesture misclassifications.
        hand_type_misclassifications: a list of hand type misclassifications.

    """

    sep = '/'

    # Get unique merged text labels.
    unique_text_labels = set()

    hand_type_right_classified = set()
    [hand_type_right_classified.add(i) for i in range(len(gt_labels))]
    hand_type_right_classified = \
        hand_type_right_classified.difference(hand_type_misclassifications)
    hand_type_right_classified = list(hand_type_right_classified)

    for i in hand_type_right_classified:
        label = gt_labels[i]
        unique_text_labels.add(label[0] + sep + label[1])

    for i in hand_type_right_classified:
        label = pr_labels[i]
        unique_text_labels.add(label[0] + sep + label[1])

    unique_text_labels = sorted(list(unique_text_labels))

    # Get from text label to integer value mapping.
    new_text_to_label = {}
    for i in range(len(unique_text_labels)):
        new_text_to_label[unique_text_labels[i]] = i

    # Compute confusion matrix and save it to file.
    new_gt_labels = []
    new_pr_labels = []
    for i in hand_type_right_classified:
        new_gt_labels.append(gt_labels[i][0] + sep + gt_labels[i][1])
        new_pr_labels.append(pr_labels[i][0] + sep + pr_labels[i][1])

    # cm = get_conf_mat_and_misclassifications(new_gt_labels, new_pr_labels,
    #                                          new_text_to_label,
    #                                          misclassifications)

    cm = get_conf_mat_and_misclassifications_non_norm(new_gt_labels, new_pr_labels,
                                                      new_text_to_label,
                                                      misclassifications)

    # Prepare unique gestures
    unique_gestures = []
    for label in unique_text_labels:
        slash_pos = label.find('/')
        left_gesture = label[:slash_pos]
        right_gesture = label[slash_pos+1:]
        should_add_left = True
        should_add_right = True
        for gesture in unique_gestures:
            if gesture == left_gesture:
                should_add_left = False
            if gesture == right_gesture:
                should_add_right = False
        if should_add_left:
            unique_gestures.append(left_gesture)
        if should_add_right and (not right_gesture == left_gesture):
            unique_gestures.append(right_gesture)

    # remove non target combinations
    temp_cm = cm
    col_both_hand = np.zeros(len(unique_text_labels))
    temp_labels = unique_text_labels
    should_cut = True
    while should_cut:
        rows, cols = temp_cm.shape
        one_col_cuted = False
        col = 0
        while (not one_col_cuted) and (col < cols):
            label = temp_labels[col]
            slh = label.find('/')
            left = label[:slh]
            right = label[slh+1:]
            if not ((left == 'no_hand' and not right == 'no_hand') or (right == 'no_hand' and not left == 'no_hand')):
                for c in range(cols):
                    col_both_hand[c] += temp_cm[c][col]

                col_both_hand = np.delete(col_both_hand, col)
                temp_labels = np.delete(temp_labels, col)
                temp_cm = np.delete(temp_cm, col, 0)
                temp_cm = np.delete(temp_cm, np.s_[col], 1)

                one_col_cuted = True
            should_cut = one_col_cuted
            col += 1



    #merge correspondent labels
    # rows, cols = temp_cm.shape
    # temp_col_both_hands = col_both_hand
    # new_labels = temp_labels
    # col = 0
    # while col < cols:
    #     sh = new_labels[col].find('/')
    #     left = new_labels[col][:sh]
    #     right = new_labels[col][sh+1:]
    #
    #     gesture = ''
    #     if left == 'no_hand':
    #         gesture = right
    #     else:
    #         gesture = left
    #
    #     pair = -1
    #     for c in range(col + 1, cols):
    #         idx = new_labels[c].find(gesture)
    #         if  not idx == -1:
    #             pair = c
    #
    #     if not pair == -1:
    #         for r in range(rows):
    #             if not r == pair:
    #                 temp_col_both_hands[r] += temp_cm[r][pair]
    #         temp_cm[col][col] += temp_cm[pair][pair]
    #         temp_cm = np.delete(temp_cm, np.s_[pair], 1)
    #         rows, cols = temp_cm.shape
    #
    #         new_labels = np.delete(new_labels, pair)
    #         new_labels[col] = gesture
    #
    #         for c in range(cols):
    #             devid = new_labels[c].find('/')
    #             if devid != -1:
    #                 cur_left = new_labels[c][:devid]
    #                 cur_rite = new_labels[c][devid+1:]
    #
    #                 find_consistent_label = cur_rite + '/'+ cur_left
    #
    #                 for k in range(len(new_labels)):
    #                     if find_consistent_label == new_labels[k]:
    #                         temp_cm[col][k] += temp_cm[pair][c]
    #             else:
    #                 #  if this is avg column this should add
    #                 temp_cm[col][c] += temp_cm[pair][c]
    #         temp_cm = np.delete(temp_cm, pair, 0)
    #         rows, cols = temp_cm.shape
    #
    #         temp_col_both_hands[col] += temp_col_both_hands[pair]
    #         temp_col_both_hands = np.delete(temp_col_both_hands, pair)
    #     else:
    #         new_labels[col] = gesture
    #
    #     rows, cols = temp_cm.shape
    #     col += 1

    rows, cols = temp_cm.shape

    avg_matrix = np.zeros((rows, len(unique_gestures) + 1), dtype=np.float)
    rows_avg, cols_avg = avg_matrix.shape

    row_labels = []
    for i in range(len(temp_labels)):
        sh = temp_labels[i].find('/')
        left = temp_labels[i][:sh]
        right = temp_labels[i][sh+1:]
        if left != 'no_hand':
            gesture = left
        else:
            gesture = right
        row_labels.append(gesture)

    last_col = cols_avg - 1

    for i in range(rows):
        cur_label = temp_labels[i]
        sh = cur_label.find('/')
        left = cur_label[:sh]
        right = cur_label[sh+1:]
        gesture = ''
        rg = False
        if left == 'no_hand':
            gesture = right
            rg = True
        else:
            rg = False
            gesture = left

        label_pattern = ''
        if rg:
            label_pattern = 'no_hand/'
        else:
            label_pattern = '/no_hand'

        avg_matrix[i][last_col] += col_both_hand[i]
        for j in range(cols):
            ptr_pos = temp_labels[j].find(label_pattern)
            if ptr_pos == -1:
                avg_matrix[i][last_col] += temp_cm[i][j]
            else:
                temp_sh_pos = temp_labels[j].find('/')
                temp_left = temp_labels[j][:temp_sh_pos]
                temp_right = temp_labels[j][temp_sh_pos+1:]
                temp_gesture = ''
                if rg:
                    temp_gesture = temp_right
                else:
                    temp_gesture = temp_left

                for k in range(cols_avg - 1):
                    if unique_gestures[k] == temp_gesture:
                        avg_matrix[i][k] += temp_cm[i][j]

    # shrink rows
    rows, cols = avg_matrix.shape
    for i in range(rows):
        j = i+1
        should_find = True
        while should_find and (j < rows):
            if row_labels[i] == row_labels[j]:
                should_find = False
                for k in range(cols):
                    avg_matrix[i][k] += avg_matrix[j][k]
                avg_matrix = np.delete(avg_matrix, j, axis=0)
                row_labels = np.delete(row_labels, j)
                rows, cols = avg_matrix.shape
            j+=1
    nh_del_id = 0
    for i in range(len(unique_gestures)):
        if unique_gestures[i] == 'no_hand':
            nh_del_id = i
    unique_gestures = np.delete(unique_gestures, nh_del_id)
    avg_matrix = np.delete(avg_matrix, np.s_[nh_del_id], axis=1)
    rows, cols = avg_matrix.shape

    temp_col_both_hands = []
    for i in range(rows):
        temp_col_both_hands.append(avg_matrix[i][last_col - 1])
    avg_matrix = np.delete(avg_matrix, np.s_[last_col-1], axis=1)

      ### reorder columns
    rows, cols = avg_matrix.shape
    reordered_matrix = np.zeros((rows, rows), dtype=np.float)
    for i in range(rows):
        row_gest = row_labels[i]
        for j in range(cols):
            if unique_gestures[j] == row_gest:
                for k in range(rows):
                    reordered_matrix[k][i] = avg_matrix[k][j]

    avg_matrix = copy(reordered_matrix)

    temp_cm = copy(avg_matrix)
    new_labels = copy(row_labels)
    rows, cols = avg_matrix.shape

    # represent matrix as binary task gesture vs none
    none_ids = -1
    # for i in range(len(unique_gestures)):
    #     if unique_gestures[i] == 'none':
    #         none_ids = i
    # for i in range(len(unique_gestures)):
    #     for j in range(len(unique_gestures)):
    #         if j != none_ids and i != j:
    #             temp_cm[i][none_ids] += temp_cm[i][j]
    #             temp_cm[i][j] = 0

    temp_cm_norm = temp_cm
    for row in range(cols):
        sm = 0
        for col in range(cols):
            sm += temp_cm[row][col]

        for col in range(cols):
            temp_cm_norm[row][col] = (100.0*temp_cm[row][col])/sm

    temp_cm_norm = np.round(temp_cm_norm, 3)

    for i in range(len(misclassifications)):
        misclassifications[i] = \
            hand_type_right_classified[misclassifications[i]]

    plot_confusion_matrix(temp_cm_norm, new_labels, "Hand pose conf mat")
    plt.savefig(path + "_cm.png")
    plt.clf()

    # Compute histograms for merged text labels.
    # right_hist_start_idx = len(text_to_label)
    # new_estimated_hists = []
    # for i in hand_type_right_classified:
    #     hist = estimated_hists[i]
    #     new_hist = []
    #     for label in unique_text_labels:
    #         label = label.split(sep)
    #         h = min(hist[text_to_label[label[0]]],
    #                 hist[right_hist_start_idx + text_to_label[label[1]]])
    #         new_hist.append(h)
    #     hist_sum = sum(new_hist)
    #     if hist_sum > 0:
    #         new_hist = [x / hist_sum for x in new_hist]
    #     new_estimated_hists.append(new_hist)

    new_gt_labels_int = []
    for label in new_gt_labels:
        new_gt_labels_int.append(new_text_to_label[label])

        #plot_precision_recall_curves(new_gt_labels_int, new_text_to_label.keys(),
        #new_estimated_hists, "Precision recall")
    plt.savefig(path + "_pr.png")
    plt.clf()


def collect_mean_values(hand_idx, gt_labels, pr_labels, gt_joints, pr_joints,
                        errors_per_joint, check_gesture_equality):
    """Collects regression errors for each joint.

    Args:
        hand_idx: index of hand
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        gt_joints: ground truth joints list.
        pr_joints: predicted joints list.
        errors_per_joint: keeps errors (L2) for each joints separately.
        check_gesture_equality: it keeps true if gesture equality check is on.

    """

    compared = 0
    total = 0
    for i in xrange(len(gt_labels)):
        gt_label = gt_labels[i][hand_idx]
        pr_label = pr_labels[i][hand_idx]
        gt_joint = gt_joints[i][hand_idx]
        pr_joint = pr_joints[i][hand_idx]
        if len(gt_joint) > 0:
            total += 1
        if ((not check_gesture_equality) or gt_label == pr_label) and \
                        len(gt_joint) > 0 and len(gt_joint) == len(pr_joint):
            errors_per_joint[gt_label][0].append(
                distance.euclidean(gt_joint[0], pr_joint[0]))
            errors_per_joint[gt_label][1].append(
                distance.euclidean(gt_joint[1], pr_joint[1]))
            compared += 1
    return compared, total


def save_joints_regression_result(path, gt_labels, pr_labels, gt_joints,
                                  pr_joints, text_to_label,
                                  check_gesture_equality):
    """Saves violin plots for each joint separately.

    Args:
        path: where plots will be saved to.
        gt_labels: ground truth labels list.
        pr_labels: predicted labels list.
        gt_joints: ground truth joints list.
        pr_joints: predicted joints list.
        text_to_label: a map from a text label to an integer value.
        check_gesture_equality: it keeps true if gesture equality check is on.

    Returns:
        Fraction of frames compared with ground truth.

    """

    labels = text_to_label.keys()
    errors_per_joint = dict()
    for label in labels:
        errors_per_joint[label] = [[], []]

    compared = 0
    total = 0
    stats = collect_mean_values(0, gt_labels, pr_labels, gt_joints,
                                pr_joints, errors_per_joint,
                                check_gesture_equality)
    compared += stats[0]
    total += stats[1]
    stats = collect_mean_values(1, gt_labels, pr_labels, gt_joints,
                                pr_joints, errors_per_joint,
                                check_gesture_equality)
    compared += stats[0]
    total += stats[1]

    could_compare = 0
    if total > 0:
        could_compare = float(compared) / total

    if could_compare > 0:
        valid_joints = 0
        for label in errors_per_joint:
            if len(errors_per_joint[label][0]) > 0 and \
                            len(errors_per_joint[label][1]) > 0:
                valid_joints += 1

        dpi = 140
        size = (1024, 720)
        font_size = 8

        plt.figure(figsize=(size[0] / dpi, size[1] / dpi))
        matplotlib.rc('font', size=font_size)

        figure = plt.gcf()

        figure.text(0.5, 0, "Compared " + "{:.2f} ".format(could_compare * 100)
                    + "% frames", horizontalalignment='center',
                    bbox=dict(facecolor='red', alpha=0.5),
                    verticalalignment='bottom')

        cols = 3
        rows = valid_joints / cols
        if valid_joints % cols != 0:
            rows += 1

        x = 1
        for label in errors_per_joint:
            if len(errors_per_joint[label][0]) > 0 and \
                            len(errors_per_joint[label][1]) > 0:
                plot = figure.add_subplot(rows, cols, x)
                data = [errors_per_joint[label][0], errors_per_joint[label][1]]
                sm.graphics.violinplot(data, ax=plot, show_boxplot=True)
                plt.scatter(
                    [i for i in range(1, len(errors_per_joint[label]) + 1)],
                    np.mean(errors_per_joint[label], axis=1))

                plt.xlabel(label)
                plt.grid(True)
                plt.ylim(0.0, 0.1)
                plt.ylabel('Distance error, m')
                x += 1
        plt.tight_layout(pad=2)
        plt.savefig(path, dpi=dpi)
        plt.clf()
    return could_compare > 0


def collect_errors_for_centers(hand_idx, gt_joints, pr_joints, errors):
    """Collects regression errors for hand centers.

    Args:
        hand_idx: index of hand
        gt_joints: ground truth joints list.
        pr_joints: predicted joints list.
        errors: keeps errors (L2) for each hanc center.

    """

    pos = 0
    for i in xrange(len(gt_joints)):
        gt_joint = gt_joints[i][hand_idx]
        pr_joint = pr_joints[i][hand_idx]
        if len(gt_joint) > 0 and len(gt_joint) == len(pr_joint):
            errors.append(distance.euclidean(gt_joint, pr_joint))
            pos += 1
    return pos


def save_joints_regression_result_centers(path, gt_joints, pr_joints):
    """Saves violin plots for each hand center separately.

    Args:
        path: where plots will be saved to.
        gt_joints: ground truth joints list.
        pr_joints: predicted joints list.

    Returns:
        Fraction of frames compared with ground truth.

    """

    left_center_error = []
    right_center_error = []

    could_compare = 0.0
    could_compare_left = \
        collect_errors_for_centers(0, gt_joints, pr_joints, left_center_error)
    could_compare += float(could_compare_left) / len(gt_joints)
    could_compare_right = \
        collect_errors_for_centers(1, gt_joints, pr_joints, right_center_error)
    could_compare += float(could_compare_right) / len(gt_joints)

    if could_compare > 0:
        x = 1
        dpi = 140
        size = (1024, 720)
        font_size = 8

        plt.figure(figsize=(size[0] / dpi, size[1] / dpi))
        matplotlib.rc('font', size=font_size)

        figure = plt.gcf()

        figure.text(0.5, 0, "Compared " + "{:.2f}".format(could_compare * 100)
                    + "% frames", horizontalalignment='center',
                    bbox=dict(facecolor='red', alpha=0.5),
                    verticalalignment='bottom')

        plot = figure.add_subplot(1, 1, 1)
        labels = []
        data = []
        means = []
        if could_compare_left:
            labels.append("left")
            data.append(left_center_error)
            means.append(np.mean(left_center_error))
        if could_compare_right:
            labels.append("right")
            data.append(right_center_error)
            means.append(np.mean(right_center_error))

        sm.graphics.violinplot(data, ax=plot, show_boxplot=True)
        plt.scatter([i for i in range(1, len(means) + 1)], means)
        plot.set_xticklabels(labels)
        plt.xlabel("center")
        plt.grid(True)
        plt.ylim(0)
        plt.ylabel('Distance error, m')
        x += 1

        plt.tight_layout(pad=2)
        plt.savefig(path, dpi=dpi)
        plt.clf()

    return could_compare > 0


def load_joints(path):
    """Loads 3D joints from disk.

    Args:
        path: where joints will be loaded from.

    Returns:
        A list of lists. Each list contains 3D joints for left and right hand.

    """
    joints = []
    with open(path, "r") as f:
        content = f.readlines()
        for line in content:
            line = [float(x) for x in line.strip().split(" ")]
            i = 0
            lh_joints = []
            rh_joints = []
            lh_joints_len = int(line[i])
            i += 1
            for j in xrange(lh_joints_len):
                lh_joints.append([line[i], line[i + 1], line[i + 2]])
                i += 3
            rh_joints_len = int(line[i])
            i += 1
            for j in xrange(rh_joints_len):
                rh_joints.append([line[i], line[i + 1], line[i + 2]])
                i += 3
            joints.append([lh_joints, rh_joints])
    return joints


id_counter = 0


def add_button(name, image):
    """Adds a button to a html page.

    Args:
        name: button's name.
        image: related image.

    Returns:
        A html text for button.

    """
    global id_counter
    id = "b" + str(id_counter)
    id_counter += 1
    image_id = image.replace(".", "_")
    image_id = image_id.replace("/", "_")

    text = "        <button id=\"" + id + \
           "\" title=\"Click to show content\"type=\"button\" " \
           "onclick=\"ShowHide(this.id, '" + image_id + "');\" >" + \
           name + "</button>\n" \
                  "        <img id=\"" + image_id + "\" title=" + name + \
           " (click to hide)\" " \
           "align=\"middle\" src=" + image + \
           " onclick=\"ShowHide('" + id + "', this.id);\" " \
                                          "style=\"display:none;\" />\n" \
                                          "        <br>\n\n\n"

    return text


def make_html_page(path, add_joints_whole, add_joints, add_centers,
                   description, add_mispredictions):
    """Creates a html page that contains quality report.

    Args:
        path: where a page will be saved to.
        add_joints_whole: it keeps true if joints regression quality is required
                          for whole pipeline.
        add_joints: it keeps true if joints regression quality is required.
        add_centers: it keeps true if hand centers regression quality is
                     required.

    Returns:
        A html text for report.

    """

    html_text = "<!DOCTYPE html>\n" \
                "<html>\n" \
                "<head>\n" \
                "<meta charset=\"utf-8\" />\n" \
                "<title>classifier errors</title>\n" \
                "\n" \
                "<script src=\"d3.min.js\" charset=\"utf-8\"></script>\n" \
                "<script type=\"text/javascript\">\n" \
                "function ShowHide(buttonid, elementid) {\n" \
                "    var element = document.getElementById(elementid);\n" \
                "    var button = document.getElementById(buttonid);\n" \
                "    if (element.style.display == \"none\") {\n" \
                "        element.style.display = \"\";" \
                "        button.style.visibility = \"hidden\"\n" \
                "    } else {\n" \
                "        element.style.display = \"none\";\n" \
                "        button.style.visibility = \"visible\"\n" \
                "    }\n" \
                "}\n" \
                "</script>\n" \
                "\n" \
                "\n" \
                "</head>\n" \
                "<body>\n" \
                "<div class=\"wrap\">\n" \
                "    <div align=\"center\">\n" \
                "        <h1>Random Forest-based pipeline quality</h1>\n" \
                "    </div>\n" \
                "\n" \
                "    <div align=\"center\">\n"

    html_text += """

<style>

.hist_place {

}

.bar {
  font-size: 10px;
}

.bar rect {
  fill: steelblue;
  shape-rendering: crispEdges;
}

.bar text {
  fill: #fff;
}

.axis path, .axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

</style>
<script>

function histogramm(labels, counts, tag) {
    var values = d3.range(1000).map(d3.random.bates(10));

    var fake = [];
    var ticks = []
    var j = 0;
    for (var i = 0; i < labels.length; i++){
        ticks[i] = i + 0.25;
        console.log(parseInt(counts[i]));
        for (var k = 0; k < parseInt(counts[i]); k++)
        {
            fake[j] = i;
            j++;
        }


    }

    var formatCount = d3.format(",.0f");
    var margin = {top: 10, right: 30, bottom: 30, left: 30},
    width = 900 - margin.left - margin.right,
    height = 300 - margin.top - margin.bottom;

    var x = d3.scale.linear()
    .domain([0, labels.length])
    .range([0, width]);

    var data = d3.layout.histogram()
    .bins(x.ticks(20))
    (fake);

    var y = d3.scale.linear()
    .domain([0, d3.max(data, function(d) { return d.y; })])
    .range([height, 0]);

    var xAxis = d3.svg.axis()
    .scale(x)
    .tickValues(ticks)
    .tickFormat(function(d) { return labels[d-0.25]; })
    .orient("bottom");

    var svg = d3.select(tag).append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var bar = svg.selectAll(".bar")
        .data(data)
        .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function(d) { return "translate(" + x(d.x) + "," + y(d.y) + ")"; });

    bar.append("rect")
        .attr("x", 1)
        .attr("width", x(data[0].dx) - 1)
        .attr("height", function(d) { return height - y(d.y); });

    bar.append("text")
        .attr("dy", ".75em")
        .attr("y", 6)
        .attr("x", x(data[0].dx) / 2)
        .attr("text-anchor", "middle")
        .text(function(d) { return formatCount(d.y); });

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);
}

</script>

    """

    html_text += description + '\n'

    html_text +="<h1>Whole pipeline results</h1>\n\n"

    html_text += add_button("Confusion matrix", "img/whole_cm.png")
    html_text += add_button("Precision recall curves", "img/whole_pr.png")
    if add_mispredictions:
        html_text += "<a href=\"whole_mis.html\">Mispredictions</a>\n<br>\n"

    if add_joints_whole:
        html_text += add_button("Joints regression", "img/joints_whole.png")

    html_text += "        <h1>Separate parts of pipeline</h1>\n\n"

    html_text += add_button("Confusion matrix", "img/hand_pose_cm.png")
    if add_mispredictions:
        html_text += "<a href=\"hand_pose_mis.html\">Mispredictions</a>\n<br>\n"
    html_text += add_button("Precision recall curves", "img/hand_pose_pr.png")
    html_text += add_button("Hand type", "img/hand_type_cm.png")
    if add_mispredictions:
        html_text += "<a href=\"hand_type_mis.html\">Mispredictions</a>\n<br>\n"

    if add_joints:
        html_text += add_button("Joints regression", "img/joints.png")
    if add_centers:
        html_text += add_button("Centers regression", "img/centers.png")

    html_text += "\n</div>\n\n</div>\n</body>\n</html>\n"

    with open(path, "w") as f:
        f.write(html_text)


def parse_args():
    """Parses arguments of command line.

    Returns:
        Command line arguments.

    """

    class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=ArgsFormatter,
                                     description="Draws precision-recall curves"
                                                 "and builds confusion matrix.")

    parser.add_argument("--input", required=True,
                        help="Where input datasets are stored.")

    parser.add_argument("--add_mispredictions", action="store_true",
                        help="Add mispredictions to report or not.")

    return parser.parse_args()


def get_hl_models(xml_node):
    """Gets path to hand pose classifier and labels from graph file.

    Returns:
        Hand pose model path, hand pose labels path.
    """

    model_path = ""
    labels_path = ""
    for a in xml_node.findall('set'):
        if a.get('param') == "model_path":
            model_path = a.get('value')
        if a.get('param') == "labels_path":
            labels_path = a.get('value')
    return model_path, labels_path


def get_segm_models(xml_node):
    """Gets segmentation models info.

    Gets path to hand center model and left-right classifier
    from graph file.

    Returns:
        Hand center model path, left-right classifier model path.
    """

    model_hc_path = ""
    model_lr_path = ""
    for a in xml_node.findall('set'):
        if a.get('param') == "model_segm_hc":
            model_hc_path = a.get('value')
        if a.get('param') == "model_segm_lr":
            model_lr_path = a.get('value')
    return model_hc_path, model_lr_path


def get_keypoints_models(xml_node):
    """Gets path to keypoints models.

    Returns:
        Keypoints models folder, keypoints models config.
    """

    model_folder = ""
    config = ""
    for a in xml_node.findall('set'):
        if a.get('param') == "model_folder":
            model_folder = a.get('value')
        if a.get('param') == "config":
            config = a.get('value')
    return model_folder, config


def parse_model_names(graph_file, output_dir):
    """Gets paths all models in graph.

    Returns:
        Pipeline models paths.
    """

    copied_xml = \
        os.path.join(output_dir, os.path.basename(graph_file) + ".copy.xml")
    with open(graph_file, "r") as f:
        content = f.readlines()
        with open(copied_xml, "w") as f1:
            f1.write("<root>\n")
            for line in content:
                f1.write(line)
            f1.write("</root>\n")

    e = xml.etree.ElementTree.parse(copied_xml).getroot()
    included = ""
    for atype in e.findall('include'):
        included = str(atype.get('file')).replace("${SCRIPT_START}",
                                                  os.getenv("SCRIPT_START"))

    with open(graph_file, "r") as f:
        content = f.readlines()
        with open(copied_xml, "w") as f1:
            f1.write("<root>\n")
            if included != "None" and included != "":
                with open(included, "r") as f2:
                    content += f2.readlines()
            f1.writelines(content)
            f1.write("</root>\n")

    e = xml.etree.ElementTree.parse(copied_xml).getroot()

    model_hl = ""
    labels_hl = ""
    model_hc = ""
    model_lr = ""
    keypoints_models_folder = ""
    keypoints_models_config = ""

    for a in e.findall('mlx'):
        for b in a.findall('nodes'):
            for c in b.findall('create'):
                if c.get('type') == "NodeClsHighlevel":
                    model_hl, labels_hl = get_hl_models(c)
                if c.get('type') == "NodeHandsSegmentation":
                    model_hc, model_lr = get_segm_models(c)
                if c.get('type') == "NodeGestureKeypoints":
                    keypoints_models_folder, keypoints_models_config = \
                        get_keypoints_models(c)

    return model_hl, labels_hl, model_hc, model_lr, \
           keypoints_models_folder, keypoints_models_config


def get_html_table_row_th_td(th, td):
    """Returns a row as html text."""

    return "<tr> <th> " + th + "</th> <td> " + td + "</td> </tr> \n"


def get_html_table_row_th_th(th1, th2):
    """Returns a row as html text."""

    return "<tr> <th> " + th1 + "</th> <th> " + th2 + "</th> </tr> \n"


def get_html_table_head(header):
    """Returns a table header as html text."""

    html = "<thead>\n" + get_html_table_row_th_th('', header) + " </thead>\n"
    return html


def html_description(timestamp, model_hl, labels_hl, model_hc, model_lr,
                     keypoints_models_folder, keypoints_models_config):
    """Returns experiment description as html text."""

    cur_date = timestamp[4:8] + "/" + timestamp[2:4] + "/" + timestamp[0:2]
    cur_time = timestamp[9:11] + ":" + timestamp[11:13] + ":" + timestamp[13:15]

    html_text = "<table border=\"1\" class=\"fixed\">\n" + \
                get_html_table_head("Date") + \
                "<tbody>\n" + \
                get_html_table_row_th_td("Date", cur_date) + \
                get_html_table_row_th_td("Time", cur_time) + \
                "</tbody>\n" \
                "</table><br>\n"
    if model_hl != "":
        html_text += "<table border=\"1\" class=\"fixed\">\n" + \
                     get_html_table_head("NodeClsHiglevel") + \
                     "<tbody>\n" + \
                     get_html_table_row_th_td("Model", model_hl) + \
                     get_html_table_row_th_td("Labels", labels_hl) + \
                     "</tbody>\n" \
                     "</table><br>\n"
    if model_hc != "":
        html_text += "<table border=\"1\" class=\"fixed\">\n" + \
                     get_html_table_head("NodeHandsSegmentation") + \
                     "<tbody>\n" + \
                     get_html_table_row_th_td("Hand center model", model_hc) + \
                     get_html_table_row_th_td("Left-right classifier",
                                              model_lr) + \
                     "</tbody>\n" \
                     "</table><br>\n"
    if keypoints_models_folder != "":
        html_text += "<table border=\"1\" class=\"fixed\">\n" + \
                     get_html_table_head("NodeGestureKeypoints") + \
                     "<tbody>\n" + \
                     get_html_table_row_th_td("Models folder",
                                              keypoints_models_folder) + \
                     get_html_table_row_th_td("Models config",
                                              keypoints_models_config) + \
                     "</tbody>\n" \
                     "</table><br>\n"

    return html_text


def print_mispredictions(paths, mispredictions, gt_labels, pr_labels,
                         images_folder, output_file, dataset_folder):
    """Returns mispredictions as a part of as html page."""

    saved = 0
    size_pix_w = 800
    size_pix_h = 240
    dpi = 80

    os.system("mkdir -p " + images_folder)

    images_html = "<!DOCTYPE html>\n" \
                  "<html>\n" \
                  "<head>\n" \
                  "<meta charset=\"utf-8\" />\n" \
                  "<title>classifier errors</title>\n" \
                  "</head>\n" \
                  "<body>\n" \
                  "<div align=\"center\">" \
                  "</div>\n" \
                  "</body>\n" \
                  "</html>"

    for i in mispredictions:
        path = paths[i]
        plt.figure(figsize=[size_pix_w / dpi, size_pix_h / dpi], dpi=80)

        plt.axis('off')
        # Read image.
        im = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        # Truncate too high and too low depth values.
        im[im < 10] = 0
        im[1000 < im] = 0
        # Show depth image in HSV color map.
        color_map = matplotlib.cm.get_cmap('hsv')
        # Show zero values of depth (after processing) as gray pixels.
        color_map.set_under('0.5')
        plt.imshow(im, interpolation='none', cmap=color_map, vmin=0.01)
        plt.title("ACTUAL: " + gt_labels[i][0] + "/" + gt_labels[i][1] +
                  " PREDICTED: " + pr_labels[i][0] + "/" + pr_labels[i][1])

        # Dump the figure to the file.
        plt.tight_layout()
        img_path = os.path.join(images_folder, os.path.basename(output_file) +
                                str(saved) + ".png")
        relative_img_path = os.path.join("img", os.path.basename(output_file) +
                                         str(saved) + ".png")
        images_html += "<img src=" + relative_img_path;
        relative_source_path = path.replace(dataset_folder, '', 1)
        if relative_source_path[0] == '/':
            relative_source_path.replace('/', '', 1)
        images_html += " title=" + relative_source_path
        images_html += " />\n"

        try:
            plt.savefig(img_path, dpi=dpi)
        except:
            im.fill(255)
            cv2.imwrite(img_path, im)

        saved += 1

        # Clear the figure to free memory.
        plt.clf()
        plt.close()

    images_html += "</div>\n" \
                   "</body>\n" \
                   "</html>"

    with open(output_file, "w") as f:
        f.write(images_html)

def get_unique_specs(specs):
    unique_specs = []
    for sp in specs:
        was = False
        for us in  unique_specs:
            if us == sp:
                was = True
        if not was:
            unique_specs.append(sp)
    return unique_specs

def genertate_non_normed_rsume(out_file,
                            gt_labels, pr_labels,
                            specs,
                            text_to_label,
                            unique_labels,
                            unique_specs,
                            whole_pipeline_misclassifications):


    # Get unique merged text labels.
    sep = '/'
    unique_text_labels = set()

    for i in range(len(gt_labels)):
        label = gt_labels[i][0] + sep + gt_labels[i][1]
        was = False
        for l in unique_text_labels:
            if l == label:
                was = True
        if not was:
            unique_text_labels.add(label)

        label = pr_labels[i][0] + sep + pr_labels[i][1]
        was = False
        for l in unique_text_labels:
            if l == label:
                was = True
        if not was:
            unique_text_labels.add(label)


    unique_text_labels = sorted(list(unique_text_labels))

            # Get from text label to integer value mapping.
    new_text_to_label = {}
    for i in range(len(unique_text_labels)):
        new_text_to_label[unique_text_labels[i]] = i

    # Compute confusion matrix and save it to file.
    new_gt_labels = []
    new_pr_labels = []
    for i in range(len(gt_labels)):
        new_gt_labels.append(gt_labels[i][0] + sep + gt_labels[i][1])
        new_pr_labels.append(pr_labels[i][0] + sep + pr_labels[i][1])


    confusion_data = [];
    for i in range(len(unique_specs)):
        data = np.zeros((len(unique_text_labels), len(unique_text_labels)), dtype=np.int32)
        confusion_data.append(data)

    for i in range(len(gt_labels)):
        print i, len(gt_labels)
        for j in range(len(unique_specs)):
            if unique_specs[j] == specs[i]:
                print gt_labels[i]
                g_idx = new_text_to_label[new_gt_labels[i]]
                p_idx = new_text_to_label[new_pr_labels[i]]
                confusion_data[j][g_idx][p_idx] += 1

    ## print matrix to file
    out = open(out_file, 'w')
    out.write(str(len(unique_specs)) + '\n')
    out.write(str(len(unique_text_labels)) + '\n')
    for i in range(len(unique_specs)):
        out.write(unique_specs[i] + '\n')
        for j in range(len(unique_text_labels)):
            out_string = ''
            out_string += unique_text_labels[j] + ' '
            for k in range(len(unique_text_labels)):
                out_string += str(confusion_data[i][j][k]) +  ' '
            out.write(out_string + '\n')
        out.write('\n')
    out.close()


def run_evaluation_on_dataset(dataset, add_mispredictions):
    """Performs quality evaluation on dataset.

    Args:
        dataset: dataset path.
        graph_file: graphfile path.
        output_folder: otput folder.
        add_mispredictions: it keeps true if mispredictions page is required.
    """

    image_file_name_path = os.path.join(dataset, "paths.txt")
    spec_file_path = os.path.join(dataset, "spec.txt")

    expected_labels_file_path = os.path.join(dataset, "labels.txt")
    predicted_labels_file_path = os.path.join(dataset, "predicted_labels.txt")

    expected_joints_file_path = os.path.join(dataset, "joints.txt")
    predicted_joints_file_path = os.path.join(dataset, "predicted_joints.txt")

    expected_centers_file_path = os.path.join(dataset, "centers.txt")
    predicted_centers_file_path = os.path.join(dataset, "predicted_centers.txt")

    confusion_matrix_output_file_path = os.path.join(dataset,
                                                     "confusion_matrix")
    regression_metric_output_file_path = os.path.join(dataset,
                                                      "regression_metric")

    estimated_histograms_file_path = os.path.join(dataset,
                                                  "estimated_histograms.txt")

    translations_file_path = os.path.join(dataset, "translations")


    os.putenv("IMAGE_FILE_NAMES_PATH", image_file_name_path)
    os.putenv("CONFUSION_MATRIX_OUTPUT_FILE_PATH",
              confusion_matrix_output_file_path)
    os.putenv("REGRESSION_METRIC_OUTPUT_FILE_PATH",
              regression_metric_output_file_path)

    os.putenv("EXPECTED_LABELS_FILE_PATH", expected_labels_file_path)
    os.putenv("PREDICTED_LABELS_FILE_PATH", predicted_labels_file_path)

    os.putenv("ESTIMATED_HISTOGRAMS_FILE_PATH", estimated_histograms_file_path)
    os.putenv("TRANSLATIONS_FILE_PATH", translations_file_path)

    if os.path.exists(expected_joints_file_path):
        os.putenv("EXPECTED_JOINTS_FILE_PATH", expected_joints_file_path)
        os.putenv("PREDICTED_JOINTS_FILE_PATH", predicted_joints_file_path)
        shutil.copyfile(expected_joints_file_path,
                        os.path.join(dataset, "joints.txt"))
    else:
        expected_joints_file_path = ""

    if os.path.exists(expected_centers_file_path):
        os.putenv("EXPECTED_CENTERS_FILE_PATH", expected_centers_file_path)
        os.putenv("PREDICTED_CENTERS_FILE_PATH", predicted_centers_file_path)
        shutil.copyfile(expected_centers_file_path,
                        os.path.join(dataset, "centers.txt"))
    else:
        expected_centers_file_path = ""

    # Load ground truth text labels.
    gt_labels = load_labels(expected_labels_file_path)

    # Load predicted text labels.
    pr_labels = load_labels(predicted_labels_file_path)

    # Load translations.
    label_data = load_translations(translations_file_path)
    text_to_label = label_data['ttl']
    print text_to_label
    uniql = label_data['ul']

    # Load estimated histograms.
    estimated_hists = load_estimated_histograms(estimated_histograms_file_path)

    dpi = 140
    size = (1900, 1600)
    font_size = 8
    plt.figure(figsize=(size[0] / dpi, size[1] / dpi))
    matplotlib.rc('font', size=font_size)

    whole_pipeline_misclassifications = []
    plot_conf_mat_and_precision_recall(os.path.join(dataset, "img", "whole"),
                                       gt_labels, pr_labels, estimated_hists,
                                       text_to_label,
                                       whole_pipeline_misclassifications)

    hand_type_misclasifications = []
    plot_hand_type_conf_mat(os.path.join(dataset, "img", "hand_type"), gt_labels,
                            pr_labels, hand_type_misclasifications)

    hand_pose_misclassifications = []
    plot_conf_mat_and_precision_recall_hand_pose(os.path.join(dataset, "img", "hand_pose"),
                                                 gt_labels, pr_labels, estimated_hists,
                                                 text_to_label,
                                                 whole_pipeline_misclassifications)
    ## generate confusion data resume
    # load data spec
    specs = []
    with open(spec_file_path, 'r') as f:
        for line in f.readlines():
            specs.append(line.strip())

    unique_specs = get_unique_specs(specs)


    genertate_non_normed_rsume(os.path.join(dataset, "non_normed_confusion_data.txt"),
                            gt_labels,
                            pr_labels,
                            specs,
                            text_to_label,
                            uniql,
                            unique_specs,
                            whole_pipeline_misclassifications)

    if add_mispredictions:
        paths = []
        with open(image_file_name_path, "r") as f:
            for line in f.readlines():
                paths.append(line.strip())

        print_mispredictions(paths, whole_pipeline_misclassifications,
                             gt_labels, pr_labels, os.path.join(dataset, "img"),
                             os.path.join(dataset, "whole_mis.html"),
                             dataset)
        print_mispredictions(paths, hand_type_misclasifications, gt_labels,
                             pr_labels, os.path.join(dataset, "img"),
                             os.path.join(output, "hand_type_mis.html"),
                             dataset)
        print_mispredictions(paths, hand_pose_misclassifications, gt_labels,
                             pr_labels, os.path.join(dataset, "img"),
                             os.path.join(dataset, "hand_pose_mis.html"),
                             dataset)

    add_joints = False
    add_joints_whole = False
    if expected_joints_file_path != "":
        # Load ground truth joints.
        gt_joints = load_joints(expected_joints_file_path)

        # Load predicted joints.
        pr_joints = load_joints(predicted_joints_file_path)

        add_joints_whole = save_joints_regression_result(
            os.path.join(dataset, "img", "joints_whole.png"), gt_labels,
            pr_labels, gt_joints, pr_joints, text_to_label, False)
        add_joints = save_joints_regression_result(
            os.path.join(dataset, "img", "joints.png"), gt_labels, pr_labels,
            gt_joints, pr_joints, text_to_label, True)

    add_centers = False
    if expected_centers_file_path != "":
        # Load ground truth centers.
        gt_centers = load_joints(expected_centers_file_path)
        # Load predicted centers.
        pr_centers = load_joints(predicted_centers_file_path)
        # Estimate quality of center finder.
        add_centers = save_joints_regression_result_centers(
            os.path.join(dataset, "img", "centers.png"), gt_centers, pr_centers)

    description_file = open(os.path.join(dataset, 'description.html'), 'r')
    description = ""
    for line in description_file.readlines():
        description += line

    ## add buttonpress hit rate into the description if exists
    file_hit_path = os.path.join(dataset, "button_press_quality.txt")
    if os.path.exists(os.path.join(dataset, "button_press_quality.txt")):
        #tmp_txt = ""
        hit_stat = open(file_hit_path, 'r')
        watch = ''
        correct = ''
        wrong = ''
        success = ''
        first_hits = ''
        avg_time = ''
        wo_outliers = ''
        hist_bins = ''
        time_hist = ''
        for line in hit_stat.readlines():
            pos = line.find('Watch no:')
            if pos != -1:
                watch = 'no'
            else:
                pos = line.find('Watch yes')
                if pos != -1:
                    watch = 'yes'
                else:
                    pos = line.find('#correct hits: ')
                    if pos != -1:
                        correct = line[pos + len("#correct hits: "):]
                    else:
                        pos = line.find('#wrong hits: ')
                        if pos != -1:
                            wrong = line[pos + len('#wrong hits: '):]
                        else:
                            pos = line.find('success rate: ')
                            if pos != -1:
                                success = line[pos + len('success rate: '):]
                            else:
                                pos = line.find('#first hits: ')
                                if pos != -1:
                                    first_hits = line[pos + len('#first hits: '):]
                                else:
                                    pos = line.find('avg time to first hit: ')
                                    if pos != -1:
                                        avg_time = line[pos + len('avg time to first hit: '):]
                                    else:
                                        pos = line.find('w/o outliers       : ')
                                        if pos != -1:
                                            wo_outliers = line[pos + len('w/o outliers       : '):]
                                        else:
                                            pos = line.find('histogram bins: ')
                                            if pos != -1:
                                                hist_bins = line[pos + len('histogram bins: '):]
                                            else:
                                                pos = line.find('time histogram: ')
                                                if pos != -1:
                                                    time_hist = line[pos + len('time histogram: '):]


        #description += "<div style=\"white-space: pre;\">\n" + tmp_txt + "\n</div>"


        description += "<table border=\"1\" class=\"fixed\">\n" + \
                       get_html_table_head("Pressing hit rate") + \
                       "<tbody>\n" + \
                       get_html_table_row_th_td("Watch", watch) + \
                       get_html_table_row_th_td("Correct hits", correct) + \
                       get_html_table_row_th_td("Wrong hits", wrong) + \
                       get_html_table_row_th_td("Success rate", success) + \
                       get_html_table_row_th_td("First hits", first_hits) + \
                       get_html_table_row_th_td("Avg time to first hit", avg_time) + \
                       get_html_table_row_th_td("w/o outliers", wo_outliers) + \
                       "</tbody>\n" \
                       "</table><br>\n"

        description += """
                <div class=\"hist_place\" id="id1992\">
                    <h3>Time histogram</h1>
                </div>
            """
        description +=  "<script>\n histogramm(" +hist_bins + ", " + time_hist + ", '#id1992.hist_place');\n" \
                                                                                 "</script>"


    make_html_page(os.path.join(dataset, "content.html"),
                   add_joints_whole,
                   add_joints, add_centers, description,
                   add_mispredictions)



def main():
    args = parse_args()
    run_evaluation_on_dataset(args.input, args.add_mispredictions)
            #except:
             #   print "Incorrect processed dataset: " + dataset



if __name__ == "__main__":
    main()
