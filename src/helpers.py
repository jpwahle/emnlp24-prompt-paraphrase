import json

def get_task_info_from_sample():
    task_info_dict = {}
    with open('data/sample_tasks.txt', 'r') as file:
        for line in file:
            file_path = line.strip()
            try:
                with open(file_path, 'r') as task_file:
                    data = json.load(task_file)
                    source = data.get("Source", "Unknown source")
                    categories = data.get("Categories", ["Unknown category"])[0]
                    domain = data.get("Domains", "Unknown domain")
                    url = data.get("URL", "Unknown url")
                    reasoning = data.get("Reasoning", "No reasoning provided")

                    task_info_dict[file_path] = categories

                    print(f"Dataset: {file_path}")
                    print(f"Source: {source}, Category: {categories}, Domain: {domain}, URL: {url}, Reasoning: {reasoning}")

            except FileNotFoundError:
                print(f"File {file_path} not found.")

    # Store the task_info_dict in a json under data/sample_tasks_tasks_to_categories.json
    with open('data/sample_tasks_tasks_to_categories.json', 'w') as file:
        json.dump(task_info_dict, file, indent=4)

    return task_info_dict

if __name__ == '__main__':
    get_task_info_from_sample()
