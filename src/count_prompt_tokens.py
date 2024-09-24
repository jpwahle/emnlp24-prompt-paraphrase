import json
import os
import tiktoken
from statistics import mean

# Initialize the GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load the JSON file
with open("./data/sample_tasks_to_categories.json", "r") as f:
    tasks_to_categories = json.load(f)

# Initialize dictionaries to store the results
results = {}
category_results = {}

# Iterate through each task in the JSON
for task_file, category in tasks_to_categories.items():
    # Construct the full path to the task file
    full_path = os.path.join(os.getcwd(), task_file)

    # Check if the file exists
    if os.path.exists(full_path):
        with open(full_path, "r") as task_file:
            task_data = json.load(task_file)

            # Check if 'Definition' key exists and is a non-empty list
            if (
                "Definition" in task_data
                and isinstance(task_data["Definition"], list)
                and task_data["Definition"]
            ):
                # Get the first definition
                first_definition = task_data["Definition"][0]

                # Count the number of tokens
                token_count = len(tokenizer.encode(first_definition))

                # Store the result
                results[full_path] = token_count

                # Add to category results
                if category not in category_results:
                    category_results[category] = []
                category_results[category].append(token_count)
            else:
                print(f"Warning: 'Definition' not found or empty in {task_file}")
    else:
        print(f"Warning: File not found: {full_path}")

# Compute overall statistics
all_counts = list(results.values())
overall_min = min(all_counts)
overall_max = max(all_counts)
overall_avg = mean(all_counts)

# Compute statistics by category
category_stats = {}
for category, counts in category_results.items():
    category_stats[category] = {
        "min": min(counts),
        "max": max(counts),
        "avg": mean(counts),
    }

# Sort categories by average
sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]["avg"])

# Generate markdown table
markdown_table = "| Category | Min | Max | Avg |\n|----------|-----|-----|-----|\n"
for category, stats in sorted_categories:
    markdown_table += (
        f"| {category} | {stats['min']} | {stats['max']} | {stats['avg']:.2f} |\n"
    )

# Add overall statistics at the end
markdown_table += f"| Overall | {overall_min} | {overall_max} | {overall_avg:.2f} |\n"

# Print the markdown table
print(markdown_table)

# Save the markdown table to a file
with open("token_count_stats.md", "w") as f:
    f.write(markdown_table)

print(f"Total tasks processed: {len(results)}")
print(f"Statistics saved to token_count_stats.md")
