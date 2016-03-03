#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
from report_generator import load_labels
from report_generator import get_conf_mat_and_misclassifications
from repare_matrixes_one_handed import get_conf_mat_and_misclassifications_non_norm
import string
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mpld3
from mpld3 import plugins
import argparse
import xml.dom.minidom as DOM
from copy import copy

def cutXmlHeader(text):
    start = text.find('>') + 1
    out = text[start:len(text)]
    return out


class ClickInfo(mpld3.plugins.PluginBase):
    """mpld3 Plugin for getting info on click        """

    JAVASCRIPT = """
    mpld3.register_plugin("clickinfo", ClickInfo);
    ClickInfo.prototype = Object.create(mpld3.Plugin.prototype);
    ClickInfo.prototype.constructor = ClickInfo;
    ClickInfo.prototype.requiredProps = ["id"];
    ClickInfo.prototype.defaultProps = {urls:null};
    function ClickInfo(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    ClickInfo.prototype.draw = function(){
        var obj = mpld3.get_element(this.props.id);
        var urls = this.props.urls;
        obj.elements().on("mousedown",
                          function(d, i){
                            window.open(urls[i], '_blank')});
    }
    """
    def __init__(self, points, urls):
        self.points = points
        self.urls = urls
        if isinstance(points, matplotlib.lines.Line2D):
            suffix = "pts"
        else:
            suffix = None
        self.dict_ = {"type": "clickinfo",
                      "id": mpld3.utils.get_id(points, suffix),
                      "urls": urls}


def get_cm_diagonal(path_to_content):
    """Returns diagonal elements of consusion matrix computed using loaded
    content. """

    groundtruth_labels = load_labels(
        os.path.join(path_to_content, 'labels.txt'))
    predicted_labels = load_labels(
        os.path.join(path_to_content, 'predicted_labels.txt'))
    misclassifications = []

    merged_groundtruth_labels = [(l[0] + '/' + l[1]) for l in
                                 groundtruth_labels]
    merged_predicted_labels = [(l[0] + '/' + l[1]) for l in predicted_labels]
    unique_labels = list(set(merged_groundtruth_labels).union(
        merged_predicted_labels))

    text_to_label = {}
    for i in range(len(unique_labels)):
        text_to_label[unique_labels[i]] = i

    # cm = get_conf_mat_and_misclassifications(merged_groundtruth_labels,
    #                                          merged_predicted_labels,
    #                                          text_to_label, misclassifications)

    cm = get_conf_mat_and_misclassifications_non_norm(merged_groundtruth_labels,
                                             merged_predicted_labels,
                                             text_to_label, misclassifications)

    #
    # temp_cm = cm
    # should_cut_cm = True
    # while should_cut_cm:
    #     rows, cols = temp_cm.shape
    #     one_col_is_removed = False
    #     col = 0
    #     while (not one_col_is_removed) and  (col < cols):
    #         should_remove_col = True
    #         for row in range(rows):
    #             if not np.isnan(temp_cm[row][col]):
    #                 if temp_cm[row][col] > 1:
    #                     should_remove_col = False
    #         col_can_be_removed = True
    #         for c  in range(cols):
    #             if not np.isnan(temp_cm[col][c]):
    #                 col_can_be_removed = False
    #
    #         if should_remove_col and col_can_be_removed:
    #             one_col_is_removed = True
    #             temp_cm = np.delete(temp_cm, col, 0)
    #             temp_cm = np.delete(temp_cm, np.s_[col], 1)
    #
    #         col += 1
    #     should_cut_cm = one_col_is_removed
    #
    # cm = temp_cm

    print unique_labels
    unique_text_labels = unique_labels

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
    print unique_gestures

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
    print temp_labels
    print temp_cm

    #merge correspondent labels
    rows, cols = temp_cm.shape
    temp_col_both_hands = col_both_hand
    new_labels = temp_labels
    print new_labels
    col = 0
    while col < cols:
        sh = new_labels[col].find('/')
        left = new_labels[col][:sh]
        right = new_labels[col][sh+1:]

        gesture = ''
        if left == 'no_hand':
            gesture = right
        else:
            gesture = left

        pair = -1
        for c in range(col + 1, cols):
            idx = new_labels[c].find(gesture)
            if  not idx == -1:
                pair = c

        if not pair == -1:
            print pair
            for r in range(rows):
                temp_cm[r][col] += temp_cm[r][pair]
            temp_cm = np.delete(temp_cm, np.s_[pair], 1)
            rows, cols = temp_cm.shape

            for c in range(cols):
                temp_cm[col][c] += temp_cm[pair][c]
            temp_cm = np.delete(temp_cm, pair, 0)
            rows, cols = temp_cm.shape

            temp_col_both_hands[col] += temp_col_both_hands[pair]
            temp_col_both_hands = np.delete(temp_col_both_hands, pair)

            new_labels = np.delete(new_labels, pair)

        new_labels[col] = gesture
        rows, cols = temp_cm.shape
        col += 1
        print new_labels

    #print temp_cm
    print new_labels
    #print temp_col_both_hands

    temp_cm_norm = temp_cm
    for row in range(cols):
        sm = 0
        for col in range(cols):
            sm += temp_cm[row][col]
        sm += temp_col_both_hands[row]

        for col in range(cols):
            temp_cm_norm[row][col] = (100.0*temp_cm[row][col])/sm

    temp_cm_norm = np.round(temp_cm_norm, 3)

    cm = temp_cm_norm

    diagonal = []
    for i in cm.diagonal():
        if not np.isnan(i):
            diagonal.append(i)
    print diagonal
    return diagonal

def get_hit_rate(path):
    file_path = os.path.join(path, "button_press_quality.txt")
    hit_rate = "0.0"
    fild_pattern = "success rate: "
    if os.path.exists(file_path):
        data = open(file_path, 'r')
        for line in data.readlines():
            pos = line.find(fild_pattern)
            if pos != -1:
                pc_pos = line.find('%')
                hit_rate = line[pos + len(fild_pattern):pc_pos]
    return float(hit_rate)


def get_cm_diagonal_non_norm(path_to_content):
    """Returns diagonal elements of consusion matrix computed using loaded
    content. """

    groundtruth_labels = load_labels(
        os.path.join(path_to_content, 'labels.txt'))
    predicted_labels = load_labels(
        os.path.join(path_to_content, 'predicted_labels.txt'))
    misclassifications = []

    merged_groundtruth_labels = [(l[0] + '/' + l[1]) for l in
                                 groundtruth_labels]
    merged_predicted_labels = [(l[0] + '/' + l[1]) for l in predicted_labels]
    unique_labels = list(set(merged_groundtruth_labels).union(
        merged_predicted_labels))

    text_to_label = {}
    for i in range(len(unique_labels)):
        text_to_label[unique_labels[i]] = i

    # cm = get_conf_mat_and_misclassifications(merged_groundtruth_labels,
    #                                          merged_predicted_labels,
    #                                          text_to_label, misclassifications)

    cm = get_conf_mat_and_misclassifications_non_norm(merged_groundtruth_labels,
                                             merged_predicted_labels,
                                             text_to_label, misclassifications)

    #
    # temp_cm = cm
    # should_cut_cm = True
    # while should_cut_cm:
    #     rows, cols = temp_cm.shape
    #     one_col_is_removed = False
    #     col = 0
    #     while (not one_col_is_removed) and  (col < cols):
    #         should_remove_col = True
    #         for row in range(rows):
    #             if not np.isnan(temp_cm[row][col]):
    #                 if temp_cm[row][col] > 1:
    #                     should_remove_col = False
    #         col_can_be_removed = True
    #         for c  in range(cols):
    #             if not np.isnan(temp_cm[col][c]):
    #                 col_can_be_removed = False
    #
    #         if should_remove_col and col_can_be_removed:
    #             one_col_is_removed = True
    #             temp_cm = np.delete(temp_cm, col, 0)
    #             temp_cm = np.delete(temp_cm, np.s_[col], 1)
    #
    #         col += 1
    #     should_cut_cm = one_col_is_removed
    #
    # cm = temp_cm

    print unique_labels
    unique_text_labels = unique_labels

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
    print unique_gestures

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
    print temp_labels
    print temp_cm

    # #merge correspondent labels
    # rows, cols = temp_cm.shape
    # temp_col_both_hands = col_both_hand
    # new_labels = temp_labels
    # print new_labels
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
    #         print pair
    #         for r in range(rows):
    #             temp_cm[r][col] += temp_cm[r][pair]
    #         temp_cm = np.delete(temp_cm, np.s_[pair], 1)
    #         rows, cols = temp_cm.shape
    #
    #         for c in range(cols):
    #             temp_cm[col][c] += temp_cm[pair][c]
    #         temp_cm = np.delete(temp_cm, pair, 0)
    #         rows, cols = temp_cm.shape
    #
    #         temp_col_both_hands[col] += temp_col_both_hands[pair]
    #         temp_col_both_hands = np.delete(temp_col_both_hands, pair)
    #
    #         new_labels = np.delete(new_labels, pair)
    #
    #     new_labels[col] = gesture
    #     rows, cols = temp_cm.shape
    #     col += 1
    #     print new_labels
    #
    # print "IM HERE"
    # #print temp_cm
    # print new_labels
    #print temp_col_both_hands


    rows, cols = temp_cm.shape

    avg_matrix = np.zeros((rows, len(unique_gestures) + 1), dtype=np.float)
    print "This is zeros average matrix"
    print avg_matrix.shape
    rows_avg, cols_avg = avg_matrix.shape
    print avg_matrix

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
    print row_labels

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
    print "Accumulated:"
    print avg_matrix

    # shrink rows
    print "Alrm!!!!!!!!!!!!!!!!!!!"
    print row_labels
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

    print "Alrm!!!!!!!!!!!!!!!!!!!"
    print row_labels
    print avg_matrix.shape
    print unique_gestures

    print "Shrinked"
    print avg_matrix
    temp_col_both_hands = []
    for i in range(rows):
        temp_col_both_hands.append(avg_matrix[i][last_col - 1])
    print temp_col_both_hands
    avg_matrix = np.delete(avg_matrix, np.s_[last_col-1], axis=1)
    print avg_matrix.shape

    print row_labels
    print unique_gestures

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
    new_labels = copy(unique_gestures)
    rows, cols = avg_matrix.shape


    temp_cm_norm = temp_cm
    number_of_samples = []
    for row in range(cols):
        sm = 0
        for col in range(cols):
            sm += temp_cm[row][col]
        sm += temp_col_both_hands[row]

        number_of_samples.append(sm)

        for col in range(cols):
            temp_cm_norm[row][col] = (100.0*temp_cm[row][col])/sm

    temp_cm_norm = np.round(temp_cm_norm, 3)

    cm = temp_cm_norm

    print 'IM HERE'

    diagonal = []
    samples_num = []
    rows, cols = cm.shape
    for i in range(cols):
        if not np.isnan(cm[i][i]):
            diagonal.append(cm[i][i])
            samples_num.append(number_of_samples[i])
    print diagonal
    print samples_num
    return (diagonal, samples_num)

def get_html_for_dataset(dataset):
    """Returns a html page with quality timeline for dataset."""

    css = """
    table
    {
      table-layout: fixed;
      border-collapse: collapse;
      word-break:break-all;
    }
    th
    {
      color: #000000;
      background-color: #D0D0D0;
      width:150px
    }
    td
    {
      background-color: #ffffff;
      width:700px
    }

    """

    dataset_basename = os.path.basename(dataset)
    diagonals = []
    samples_nums = []

    hit_rates = []

    timestamps = sorted(os.listdir(dataset), key=lambda x: datetime.datetime.strptime(x, '%d%m%Y_%H%M%S'))
    for timestamp in timestamps:
        diagonal, samples_num = get_cm_diagonal_non_norm(os.path.join(dataset, timestamp))
        hit_rate = get_hit_rate(os.path.join(dataset, timestamp))
        print hit_rate
        print "Diagonal ", diagonal
        print samples_num
        diagonals.append(diagonal)
        samples_nums.append(samples_num)
        hit_rates.append(hit_rate)

    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)

    ax.set_axis_bgcolor('lightblue')

    means = []
    x = []
    width = 20
    height = 7
    fig.set_size_inches(width, height)

    urls = []
    descriptions = []
    date_line = []
    time_line = []
    for t in timestamps:
        description = ''
        with open(os.path.join(dataset, t, "description.html")) as f:
            for line in f.readlines():
                description += line
        correct = string.replace(description, "<br>", "")
        description_om = DOM.parseString("<root>" + correct + "</root>")
        tbody = description_om.getElementsByTagName("tbody")[0]
        date_line.append(tbody.childNodes[1].childNodes[3].childNodes[0].nodeValue)
        time_line.append(tbody.childNodes[3].childNodes[3].childNodes[0].nodeValue)

        try:
            short_description_file = open(os.path.join(dataset, t, "description.txt"), 'r')
            short_description = ""
            for line in short_description_file.readlines():
                short_description += line
            descriptions.append("<div class=\"short-hover\">" + short_description + "</div>")
        except:
            descriptions.append(description)

        urls.append(os.path.join(dataset_basename, t, "content.html"))

    for i in range(len(diagonals)):
        means.append(np.average(diagonals[i], axis=0, weights=samples_nums[i]))
        x.append(i + 1)
        #x.append(date_line[i])


    points = ax.plot(x, means, 'o', color='b',
                     mec='k', ms=25, mew=1, alpha=.6)

    weighted_box_diagonals = []
    for i in range(len(diagonals)):
        weighted = []
        for j in range(len(diagonals[i])):
            for k in range(0,int(samples_nums[i][j])):
                weighted.append(diagonals[i][j])
        weighted_box_diagonals.append(weighted)

    ax.boxplot(weighted_box_diagonals)
    plt.ylim([0, 100])
    ax.plot(x, means)
    tooltip = plugins.PointHTMLTooltip(points[0], descriptions,
                                       voffset=10, hoffset=10, css = css)
    plugins.connect(fig, tooltip)
    plugins.connect(fig, ClickInfo(points[0], urls))


    ### Try to use DOM
    doc = DOM.parseString("<div id=\"dataset\"></div>")
    dataset_node = doc.childNodes[0]

    title = doc.createElement('div')
    title.setAttribute("id", "title")

    title_text = doc.createElement('h1')
    title_text.setAttribute("align", "left")
    txt_name = doc.createTextNode(dataset_basename)
    title_text.appendChild(txt_name)

    description_text = ""
    data_descriptions = DOM.parse(os.path.join(os.path.dirname(dataset), "data_descriptions.xml"))
    description_node = data_descriptions.getElementsByTagName(dataset_basename)[0]
    description_text = description_node.childNodes[0].nodeValue

    descr = doc.createElement('div')
    descr.setAttribute("align", "left")
    txt_description = doc.createTextNode(description_text)
    descr.appendChild(txt_description)

    title.appendChild(title_text)
    title.appendChild(descr)

    plot_placeholder = doc.createElement("div")
    plot_placeholder.setAttribute("align", "left")
    plot_placeholder.setAttribute("class", "plot_time_line")
    text_marker = doc.createTextNode("{PLOT}")
    plot_placeholder.appendChild(text_marker)

    labels_script = doc.createElement("script")
    script_text = doc.createTextNode("{LABELS_SCRIPT}")
    labels_script.appendChild(script_text)

    dataset_node.appendChild(title)
    dataset_node.appendChild(plot_placeholder)
    dataset_node.appendChild(labels_script)

    plot_placeholder = doc.createElement("div")
    plot_placeholder.setAttribute("align", "left")
    plot_placeholder.setAttribute("class", "plot_time_line")
    text_marker = doc.createTextNode("{PLOT_1}")
    plot_placeholder.appendChild(text_marker)

    labels_script = doc.createElement("script")
    script_text = doc.createTextNode("{LABELS_SCRIPT_1}")
    labels_script.appendChild(script_text)

    dataset_node.appendChild(plot_placeholder)
    dataset_node.appendChild(labels_script)

    html_string = cutXmlHeader(doc.toprettyxml())
    plot_code = mpld3.fig_to_html(fig)

     ### find plot id inside the plot code.
    id_code_marker = "div id=\"fig_"
    back_indent = len("\"fig")
    start_plot_id_pos = plot_code.find(id_code_marker) + len(id_code_marker) - back_indent
    end_plot_id_pos = plot_code.find("\"", start_plot_id_pos + 1)
    plot_id_string = plot_code[start_plot_id_pos:end_plot_id_pos]
    ###!

    ### Prepare plot code with event.
    dispatch_event_code = """
        evt = document.createEvent(\"Event\");
        evt.initEvent(\"mpld3_plot_ready\", true, true);
        evt.plot_id = \"{PLOT_ID}\"
        document.dispatchEvent(evt);
    """

    dispatch_event_code = string.replace(dispatch_event_code, "{PLOT_ID}", plot_id_string)

    start = 0
    while not start == -1:
        start = plot_code.find("mpld3.draw_figure", start)
        start = plot_code.find(");", start)
        incert_idx = start + 2
        if not start == -1:
            first_part = plot_code[:incert_idx]
            second_part = plot_code[incert_idx:]
            plot_code = first_part + dispatch_event_code + second_part
            start += len(dispatch_event_code)

    ###! Prepare plot code with event.

    #### Prepare labels corrector script.
    labels_correction_code = """
        document.addEventListener("mpld3_plot_ready", function (evt) {
            if (evt.plot_id == \"{PLOT_ID}\") {
                correctXTicksLabels(\"{PLOT_ID}\", {LABELS});
            }
        });
    """

    labels_object = "[ "
    short_date = date_line[0][3:len(date_line[i])]
    short_time = time_line[0][1:6]
    labels_object += "\"" + short_date + " " + short_time + "\""
    for i in range(1, len(date_line)):
        short_date = date_line[i][3:len(date_line[i])]
        short_time = time_line[i][1:6]
        labels_object += " , \"" + short_date + " " + short_time + "\""
    labels_object += " ]"

    labels_correction_code = string.replace(labels_correction_code, "{PLOT_ID}", plot_id_string)
    labels_correction_code = string.replace(labels_correction_code, "{LABELS}", labels_object)
    ####! Prepare labels corrector script.


    html_string = string.replace(html_string, "{PLOT}", plot_code)
    html_string = string.replace(html_string, "{LABELS_SCRIPT}", labels_correction_code)

    fig.clear()



    ### add timeline for hit rate
    should_show_rate_line = False
    for rate in hit_rates:
        if rate > 0:
            should_show_rate_line = True
    if should_show_rate_line:
        print "Im here"
        fig, ax = plt.subplots()
        ax.grid(True, alpha=0.3)

        ax.set_axis_bgcolor('lightblue')


        width = 20
        height = 7
        fig.set_size_inches(width, height)

        urls = []
        descriptions = []
        date_line = []
        time_line = []
        for t in timestamps:
            description = ''
            with open(os.path.join(dataset, t, "description.html")) as f:
                for line in f.readlines():
                    description += line
            correct = string.replace(description, "<br>", "")
            description_om = DOM.parseString("<root>" + correct + "</root>")
            tbody = description_om.getElementsByTagName("tbody")[0]
            date_line.append(tbody.childNodes[1].childNodes[3].childNodes[0].nodeValue)
            time_line.append(tbody.childNodes[3].childNodes[3].childNodes[0].nodeValue)

            try:
                short_description_file = open(os.path.join(dataset, t, "description.txt"), 'r')
                short_description = ""
                for line in short_description_file.readlines():
                    short_description += line
                descriptions.append("<div class=\"short-hover\">" + short_description + "</div>")
            except:
                descriptions.append(description)

            urls.append(os.path.join(dataset_basename, t, "content.html"))

            #x.append(date_line[i])

        print len(x)
        print len(hit_rates)

        points = ax.plot(x, hit_rates, 'o', color='b',
                     mec='k', ms=25, mew=1, alpha=.6)

        fake_diag = []
        for hit in hit_rates:
            fake_diag.append([hit])

        ax.boxplot(fake_diag)
        plt.ylim([0, 100])
        ax.plot(x, hit_rates)
        tooltip = plugins.PointHTMLTooltip(points[0], descriptions,
                                       voffset=10, hoffset=10, css = css)
        plugins.connect(fig, tooltip)
        plugins.connect(fig, ClickInfo(points[0], urls))

        ## Prepare html
            ### Try to use DOM
        #doc = DOM.parseString("<div id=\"dataset\"></div>")
        #dataset_node = doc.childNodes[0]

        #title = doc.createElement('div')
        #title.setAttribute("id", "title")

        #title_text = doc.createElement('h1')
        #title_text.setAttribute("align", "left")
        #txt_name = doc.createTextNode(dataset_basename)
        #title_text.appendChild(txt_name)

        #description_text = ""
        #data_descriptions = DOM.parse(os.path.join(os.path.dirname(dataset), "data_descriptions.xml"))
        #description_node = data_descriptions.getElementsByTagName(dataset_basename)[0]
        #description_text = description_node.childNodes[0].nodeValue

        #descr = doc.createElement('div')
        #descr.setAttribute("align", "left")
        #txt_description = doc.createTextNode(description_text)
        #descr.appendChild(txt_description)

        #title.appendChild(title_text)
        #title.appendChild(descr)

        # plot_placeholder = doc.createElement("div")
        # plot_placeholder.setAttribute("align", "left")
        # plot_placeholder.setAttribute("class", "plot_time_line")
        # text_marker = doc.createTextNode("{PLOT}")
        # plot_placeholder.appendChild(text_marker)
        #
        # labels_script = doc.createElement("script")
        # script_text = doc.createTextNode("{LABELS_SCRIPT}")
        # labels_script.appendChild(script_text)
        #
        # dataset_node.appendChild(title)
        # dataset_node.appendChild(plot_placeholder)
        # dataset_node.appendChild(labels_script)

        #html_string = cutXmlHeader(doc.toprettyxml())
        plot_code = mpld3.fig_to_html(fig)

         ### find plot id inside the plot code.
        id_code_marker = "div id=\"fig_"
        back_indent = len("\"fig")
        start_plot_id_pos = plot_code.find(id_code_marker) + len(id_code_marker) - back_indent
        end_plot_id_pos = plot_code.find("\"", start_plot_id_pos + 1)
        plot_id_string = plot_code[start_plot_id_pos:end_plot_id_pos]
        ###!

         ### Prepare plot code with event.
        dispatch_event_code = """
            evt = document.createEvent(\"Event\");
            evt.initEvent(\"mpld3_plot_ready\", true, true);
            evt.plot_id = \"{PLOT_ID}\"
            document.dispatchEvent(evt);
        """

        dispatch_event_code = string.replace(dispatch_event_code, "{PLOT_ID}", plot_id_string)

        start = 0
        while not start == -1:
            start = plot_code.find("mpld3.draw_figure", start)
            start = plot_code.find(");", start)
            incert_idx = start + 2
            if not start == -1:
                first_part = plot_code[:incert_idx]
                second_part = plot_code[incert_idx:]
                plot_code = first_part + dispatch_event_code + second_part
                start += len(dispatch_event_code)

        ###! Prepare plot code with event.

        #### Prepare labels corrector script.
        labels_correction_code = """
            document.addEventListener("mpld3_plot_ready", function (evt) {
                if (evt.plot_id == \"{PLOT_ID}\") {
                    correctXTicksLabels(\"{PLOT_ID}\", {LABELS}, \"  Pressing success rate  \");
                }
            });
        """

        labels_object = "[ "
        short_date = date_line[0][3:len(date_line[i])]
        short_time = time_line[0][1:6]
        labels_object += "\"" + short_date + " " + short_time + "\""
        for i in range(1, len(date_line)):
            short_date = date_line[i][3:len(date_line[i])]
            short_time = time_line[i][1:6]
            labels_object += " , \"" + short_date + " " + short_time + "\""
        labels_object += " ]"

        labels_correction_code = string.replace(labels_correction_code, "{PLOT_ID}", plot_id_string)
        labels_correction_code = string.replace(labels_correction_code, "{LABELS}", labels_object)
        ####! Prepare labels corrector script.


        html_string = string.replace(html_string, "{PLOT_1}", plot_code)
        html_string = string.replace(html_string, "{LABELS_SCRIPT_1}", labels_correction_code)
    else:
        html_string = string.replace(html_string, "{PLOT_1}", "")
        html_string = string.replace(html_string, "{LABELS_SCRIPT_1}", "")


    return html_string

def parse_args():
    """Parses command line arguments."""
    class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=ArgsFormatter,
                                     description="Draws precision-recall curves"
                                                 "and builds confusion matrix.")

    parser.add_argument("--input", required=True,
                        help="Where input datasets are stored.")

    return parser.parse_args()

def main():
    # try:
    args = parse_args()
    dataset_folder = args.input
    datasets = os.listdir(dataset_folder)

    html_text = "<!DOCTYPE html>\n" \
                    "<html>\n" \
                    "<head>\n" \
                    "<meta charset=\"utf-8\" />\n" \
                    "<title>RF based pipeline</title>\n" \
                    "</head>\n" \
                    "<body>\n" \
                    "<div align=\"center\">\n"

    define_labels_script = """
        <style>
            .short-hover {
                width:            200px;
                padding:          3px;
                background-color: rgba(255,255,255,0.8);
                border-radius:    5px
            }

            .axis-label {
                position: relative;
                font-size: 18pt;
                font-family: serif;
            }

        </style>

        <script>
        function correctXTicksLabels(plot_id, labels, vaxis_label) {
            if (typeof vaxis_label == 'undefined') {
                vaxis_label = \"Weighted average accuracy\";
            }
            var ploted_dataset = document.getElementById(plot_id);
            var xaxis_object = ploted_dataset.getElementsByClassName(\"mpld3-xaxis\");
            var ploted_ticks = xaxis_object[0].getElementsByClassName(\"tick\");
            for (var i = 0; i < ploted_ticks.length; i++) {
                label = ploted_ticks[i].getElementsByTagName(\"text\")[0];
                label.textContent = labels[i];
                label.setAttribute(\"transform\", \"translate(-15, 70) rotate(-90)\");
            }

            var svg_figure = ploted_dataset.getElementsByClassName('mpld3-figure')[0];
            svg_figure.setAttribute(\"height\", \"670\");

            var toolbar = ploted_dataset.getElementsByClassName('mpld3-toolbar')[0];
            toolbar.parentNode.removeChild(toolbar);

            var x_axis_label = document.createElement(\'text\');
            x_axis_label.setAttribute(\"class\", \"axis-label\");
            x_axis_label.setAttribute(\"style\", \"top: -40px; left: 770px;\");
            x_axis_label.textContent = \"Date, Time\";
            ploted_dataset.appendChild(x_axis_label);
            var y_axis_label = document.createElement('div');
            y_axis_label.setAttribute(\"class\", \"axis-label\");
            y_axis_label.setAttribute(\"style\", \"top: -443px; left: -20px; width: 350px; transform: rotate(-90deg);\");
            y_axis_label.textContent = vaxis_label;
            ploted_dataset.appendChild(y_axis_label);
        };
        </script>
        """

    html_text += define_labels_script

    for dataset in datasets:
        dataset = os.path.join(dataset_folder, dataset)
        if os.path.isdir(dataset):
            html_text += get_html_for_dataset(dataset)

    html_text += "</div>\n</body>\n</html>\n"
    # except Exception, e:
    #     print str(e)
    #     return

    with open(os.path.join(dataset_folder, "content.html"), "w") as f:
        f.write(html_text)

if __name__ == "__main__":
    main()



