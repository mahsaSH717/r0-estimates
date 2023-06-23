import pandas as pd

main_df = pd.read_excel("../../../data/raw/initial_dataset/manually_annotation_result/cord19_annotated_final.xlsx")
main_df = main_df.astype(object)
cord_uid_group = main_df.groupby('cord_uid')
main_df_new = pd.DataFrame()

for cord_uid, frame in cord_uid_group:
    print(f"for {cord_uid!r}")

    frame_answer_list_text = []
    frame_answer_list_json = []
    for index, row in frame.iterrows():
        is_investigating, disease_name, location, date, r0_value, ci_values, method = row['annotator_investigating_R0'], \
            row['human_annotated_disease_name'], row['human_annotated_location'], row['human_annotated_date'], row[
            'human_annotated_r0_value'], row['human_annotated_%CI_values'], row['human_annotated_method']

        if is_investigating == -1:
            instance_answer_text = 'unanswerable'
            instance_answer_json = 'unanswerable'
        else:
            instance_answer_text = "disease name: " + str(
                row['human_annotated_disease_name']).strip() + "\n" + "location: " + str(
                row['human_annotated_location']).strip() + "\n" + "date: " + str(
                row['human_annotated_date']).strip() + "\n" + "R0 value: " + str(
                row['human_annotated_r0_value']).strip() + "\n" + "%CI values: " + str(
                row['human_annotated_%CI_values']).strip() + "\n" + "method: " + str(
                row['human_annotated_method']).strip()

            instance_answer_json = "{\"contribution\":{\"disease name\": \"" + str(
                row['human_annotated_disease_name']).strip() + "\",\n" + "\"location\": \"" + str(
                row['human_annotated_location']).strip() + "\",\n" + "\"date\": \"" + str(
                row['human_annotated_date']).strip() + "\",\n" + "\"R0 value\": \"" + str(
                row['human_annotated_r0_value']).strip() + "\",\n" + "\"%CI values\": \"" + str(
                row['human_annotated_%CI_values']).strip() + "\",\n" + "\"method\": \"" + str(
                row['human_annotated_method']).strip() + "\"}}"

        frame_answer_list_text.append(instance_answer_text)
        frame_answer_list_json.append(instance_answer_json)

    frame['text_response'] = "\n|\n".join(frame_answer_list_text)

    if frame_answer_list_json[0] != 'unanswerable':

        group_answer_json = ",".join(frame_answer_list_json)
        frame['json_response'] = "[" + group_answer_json + "]"

    else:
        frame['json_response'] = frame_answer_list_json

    my_list = [main_df_new, frame]
    main_df_new = pd.concat(my_list)

main_df_new = main_df_new.drop_duplicates(subset='cord_uid', keep="first")
main_df_new.drop(['human_annotated_disease_name', 'human_annotated_location', 'human_annotated_date',
                  'human_annotated_r0_value', 'human_annotated_%CI_values', 'human_annotated_method'], axis=1,
                 inplace=True)
final_df = main_df_new.sort_values(by=['cord_uid'])

new_column_order = ["main_cord_uid", "cord_uid", "abstract", "title", "annotator_investigating_R0", "text_response",
                    "json_response", "publish_time", "action", "cluster_id"]

final_df = final_df.reindex(columns=new_column_order)
final_df.to_excel("../../../data/raw/initial_dataset/contribution_based_dataset/group_by_dataset.xlsx", index=False)
