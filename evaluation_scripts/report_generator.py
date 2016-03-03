#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    with open(path + ".separate") as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split(" ")
            text_to_label[line[1]] = int(line[0])
    return text_to_label


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


def plot_confusion_matrix(cm, classes_map, title="Confusion matrix",
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

    cm = get_conf_mat_and_misclassifications(new_gt_labels, new_pr_labels,
                                             new_text_to_label,
                                             misclassifications)
    print cm
    temp_cm = cm
    should_cut_cm = True
    while should_cut_cm:
        rows, cols = temp_cm.shape
        one_col_is_removed = False
        col = 0
        while (not one_col_is_removed) and  (col < cols):
            should_remove_col = True
            for row in range(rows):
                if not np.isnan(temp_cm[row][col]):
                    if temp_cm[row][col] > 1:
                        should_remove_col = False
            col_can_be_removed = True
            for c  in range(cols):
                if not np.isnan(temp_cm[col][c]):
                    col_can_be_removed = False

            if should_remove_col and col_can_be_removed:
                one_col_is_removed = True
                temp_cm = np.delete(temp_cm, col, 0)
                temp_cm = np.delete(temp_cm, np.s_[col], 1)

                tmp_unique_test_labels = []
                for c in range(len(unique_text_labels)):
                    if not c == col:
                        tmp_unique_test_labels.append(unique_text_labels[c])
                unique_text_labels = tmp_unique_test_labels
            col += 1
        should_cut_cm = one_col_is_removed
        print temp_cm

    cm = temp_cm

    for i in range(len(misclassifications)):
        misclassifications[i] = \
            hand_type_right_classified[misclassifications[i]]

    plot_confusion_matrix(cm, unique_text_labels, "Hand pose conf mat")
    plt.savefig(path + "_cm.png")
    plt.clf()

    # Compute histograms for merged text labels.
    right_hist_start_idx = len(text_to_label)
    new_estimated_hists = []
    for i in hand_type_right_classified:
        hist = estimated_hists[i]
        new_hist = []
        for label in unique_text_labels:
            label = label.split(sep)
            h = min(hist[text_to_label[label[0]]],
                    hist[right_hist_start_idx + text_to_label[label[1]]])
            new_hist.append(h)
        hist_sum = sum(new_hist)
        if hist_sum > 0:
            new_hist = [x / hist_sum for x in new_hist]
        new_estimated_hists.append(new_hist)

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
                "    <div align=\"center\">\n" \
                + description + "\n" \
                                "        <h1>Whole pipeline results</h1>\n\n"

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

    parser.add_argument("--graph_file", required=True,
                        help="Path to a graph file.")

    parser.add_argument("--input", required=True,
                        help="Where input datasets are stored.")

    parser.add_argument("--add_mispredictions", action="store_true",
                        help="Add mispredictions to report or not.")

    parser.add_argument("--output", required=True, help="Output folder.")

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


def run_evaluation_on_dataset(dataset, graph_file, output_folder,
                              add_mispredictions):
    """Performs quality evaluation on dataset.

    Args:
        dataset: dataset path.
        graph_file: graphfile path.
        output_folder: otput folder.
        add_mispredictions: it keeps true if mispredictions page is required.
    """

    timestamp = time.strftime("%d%m%Y_%H%M%S")

    output = tempfile.mkdtemp()
    os.system("mkdir -p " + output)
    print "temp folder: " + output

    image_file_name_path = os.path.join(dataset, "paths.txt")

    expected_labels_file_path = os.path.join(dataset, "labels.txt")
    predicted_labels_file_path = os.path.join(output, "predicted_labels.txt")

    expected_joints_file_path = os.path.join(dataset, "joints.txt")
    predicted_joints_file_path = os.path.join(output, "predicted_joints.txt")

    expected_centers_file_path = os.path.join(dataset, "centers.txt")
    predicted_centers_file_path = os.path.join(output, "predicted_centers.txt")

    confusion_matrix_output_file_path = os.path.join(output,
                                                     "confusion_matrix")
    regression_metric_output_file_path = os.path.join(output,
                                                      "regression_metric")

    estimated_histograms_file_path = os.path.join(output,
                                                  "estimated_histograms.txt")

    translations_file_path = os.path.join(output, "translations")

    with open(image_file_name_path, "r") as f:
        image_file_name_path = os.path.join(output, "paths.txt")
        with open(image_file_name_path, "w") as f_out:
            for line in f.readlines():
                f_out.write(os.path.join(dataset, line))

    os.putenv("IMAGE_FILE_NAMES_PATH", image_file_name_path)
    os.putenv("CONFUSION_MATRIX_OUTPUT_FILE_PATH",
              confusion_matrix_output_file_path)
    os.putenv("REGRESSION_METRIC_OUTPUT_FILE_PATH",
              regression_metric_output_file_path)

    os.putenv("EXPECTED_LABELS_FILE_PATH", expected_labels_file_path)
    os.putenv("PREDICTED_LABELS_FILE_PATH", predicted_labels_file_path)
    shutil.copyfile(expected_labels_file_path,
                    os.path.join(output, "labels.txt"))

    os.putenv("ESTIMATED_HISTOGRAMS_FILE_PATH", estimated_histograms_file_path)
    os.putenv("TRANSLATIONS_FILE_PATH", translations_file_path)

    if os.path.exists(expected_joints_file_path):
        os.putenv("EXPECTED_JOINTS_FILE_PATH", expected_joints_file_path)
        os.putenv("PREDICTED_JOINTS_FILE_PATH", predicted_joints_file_path)
        shutil.copyfile(expected_joints_file_path,
                        os.path.join(output, "joints.txt"))
    else:
        expected_joints_file_path = ""

    if os.path.exists(expected_centers_file_path):
        os.putenv("EXPECTED_CENTERS_FILE_PATH", expected_centers_file_path)
        os.putenv("PREDICTED_CENTERS_FILE_PATH", predicted_centers_file_path)
        shutil.copyfile(expected_centers_file_path,
                        os.path.join(output, "centers.txt"))
    else:
        expected_centers_file_path = ""

    os.system("${MLX_MASTER_DIR}/mlx_exec -config " + graph_file)

    # Load ground truth text labels.
    gt_labels = load_labels(expected_labels_file_path)

    # Load predicted text labels.
    pr_labels = load_labels(predicted_labels_file_path)

    # Load translations.
    text_to_label = load_translations(translations_file_path)

    # Load estimated histograms.
    estimated_hists = load_estimated_histograms(estimated_histograms_file_path)

    os.system("mkdir -p " + os.path.join(output, "img"))

    dpi = 140
    size = (1900, 1600)
    font_size = 8
    plt.figure(figsize=(size[0] / dpi, size[1] / dpi))
    matplotlib.rc('font', size=font_size)

    whole_pipeline_misclassifications = []
    plot_conf_mat_and_precision_recall(os.path.join(output, "img", "whole"),
                                       gt_labels, pr_labels, estimated_hists,
                                       text_to_label,
                                       whole_pipeline_misclassifications)

    hand_type_misclasifications = []
    plot_hand_type_conf_mat(os.path.join(output, "img", "hand_type"), gt_labels,
                            pr_labels, hand_type_misclasifications)

    hand_pose_misclassifications = []
    plot_conf_mat_and_precision_recall(os.path.join(output, "img", "hand_pose"),
                                       gt_labels, pr_labels, estimated_hists,
                                       text_to_label,
                                       hand_pose_misclassifications,
                                       hand_type_misclasifications)

    if add_mispredictions:
        paths = []
        with open(image_file_name_path, "r") as f:
            for line in f.readlines():
                paths.append(line.strip())
        print_mispredictions(paths, whole_pipeline_misclassifications,
                             gt_labels, pr_labels, os.path.join(output, "img"),
                             os.path.join(output, "whole_mis.html"),
                             dataset)
        print_mispredictions(paths, hand_type_misclasifications, gt_labels,
                             pr_labels, os.path.join(output, "img"),
                             os.path.join(output, "hand_type_mis.html"),
                             dataset)
        print_mispredictions(paths, hand_pose_misclassifications, gt_labels,
                             pr_labels, os.path.join(output, "img"),
                             os.path.join(output, "hand_pose_mis.html"),
                             dataset)

    add_joints = False
    add_joints_whole = False
    if expected_joints_file_path != "":
        # Load ground truth joints.
        gt_joints = load_joints(expected_joints_file_path)

        # Load predicted joints.
        pr_joints = load_joints(predicted_joints_file_path)

        add_joints_whole = save_joints_regression_result(
            os.path.join(output, "img", "joints_whole.png"), gt_labels,
            pr_labels, gt_joints, pr_joints, text_to_label, False)
        add_joints = save_joints_regression_result(
            os.path.join(output, "img", "joints.png"), gt_labels, pr_labels,
            gt_joints, pr_joints, text_to_label, True)

    add_centers = False
    if expected_centers_file_path != "":
        # Load ground truth centers.
        gt_centers = load_joints(expected_centers_file_path)
        # Load predicted centers.
        pr_centers = load_joints(predicted_centers_file_path)
        # Estimate quality of center finder.
        add_centers = save_joints_regression_result_centers(
            os.path.join(output, "img", "centers.png"), gt_centers, pr_centers)

    models = parse_model_names(graph_file, output)
    model_hl = models[0]
    labels_hl = models[1]
    model_hc = models[2]
    model_lr = models[3]
    keypoints_models_folder = models[4]
    keypoints_models_config = models[5]
    description = html_description(timestamp, model_hl, labels_hl, model_hc,
                                   model_lr, keypoints_models_folder,
                                   keypoints_models_config)

    with open(os.path.join(output, "description.html"), "w") as f:
        f.write(description)

    make_html_page(os.path.join(output, "content.html"),
                   add_joints_whole,
                   add_joints, add_centers, description,
                   add_mispredictions)
    try:
        shutil.copytree(output, os.path.join(output_folder, timestamp))
    except Exception, e:
        print str(e)
        shutil.rmtree(os.path.join(output_folder, timestamp))

    shutil.rmtree(output)


def main():
    args = parse_args()
    for dataset in os.listdir(args.input):
        if os.path.isdir(os.path.join(args.input, dataset)):
            input = os.path.join(args.input, dataset)
            output = os.path.join(args.output, dataset)
            #try:
            run_evaluation_on_dataset(input, args.graph_file, output,
                                      args.add_mispredictions)
            #except:
             #   print "Incorrect processed dataset: " + dataset



if __name__ == "__main__":
    main()
