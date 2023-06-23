import json
import os


def meets_best_criteria(data):
    filtered_data = [item for item in data if
                     item["metrics"].get("partial_f1s_overall", 0) >= 53

                     ]
    return len(filtered_data) > 0


def process_json_files(directory):
    paths = []  # List to store the paths
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                with open(filepath) as json_file:
                    try:
                        data = json.load(json_file)
                        if meets_best_criteria(data):
                            paths.append(os.path.dirname(filepath))  # Add the path to the list
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON file:", filepath)
                        print("Error message:", str(e))
                    except Exception as e:
                        print("Error processing JSON file:", filepath)
                        print("Error message:", str(e))
    return paths


# Specify the directory path where your JSON files are located
directory_path = "../../../experimental_results/single_templates_flan_t5_fine_tuning/text_based"

# Call the function to process the JSON files and get the paths
file_paths = process_json_files(directory_path)

# Save the paths in a text file
output_file_path = "../../../experimental_results/setup_related/text_based_best_file_paths.txt"
with open(output_file_path, "w") as file:
    for path in file_paths:
        file.write(path + "\n")

print("File paths saved to:", output_file_path)
