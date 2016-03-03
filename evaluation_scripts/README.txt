TODO
To start experiment with specific mlx configuration use script "report_generator.py".
This sript put experiment results into the "example/experimants_results" folder, in the directory corresponding this experiment.

Confusion data of this experiment should be stored in the "non_normed_confusion_data.txt" file. This is not normalized confusion matrices with whole frame labels devided by specific data configuration. Matrices stored with this formt:

<number_of_specific_data_option_sets>
<number_of_existed_bothhanded_labels>
<option_set_description>
<

not normalized confusion

>

<option_set_description>
...

--- There are also should be script which gets list of considering cases and generates normalized confusion matrixes and data for time line.
    Also this should store this data into the out folder, to show it on the web page similr to the "example/out/content.html" 
