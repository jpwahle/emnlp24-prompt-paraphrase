import os
import pandas as pd
from lexicalrichness import LexicalRichness

def get_lexical_diversity(text):
    lex = LexicalRichness(text)
    return {
        'rttr': lex.rttr,
        'maas': lex.Maas,
        'mattr': lex.mattr(window_size=10) if len(text.split()) > 20 else None,
        'mtld': lex.mtld(threshold=0.72)
    }


def get_potential_performance_gain(results_path, task_to_category_path, average_method="mean"):
    import json
    import os
    from pathlib import Path
    import pandas as pd
    from tqdm import tqdm

    # Load the task-to-category mapping
    with open(task_to_category_path, 'r') as file:
        task_to_category = json.load(file)

    # List of subdirectories (one per model)
    model_dirs = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]

    # Initialize a list to hold results
    results = []

    # Iterate through each model directory
    for model_dir in tqdm(model_dirs, desc="Processing models"):
        model_path = os.path.join(results_path, model_dir)
        task_files = [f for f in os.listdir(model_path) if f.endswith(".csv")]

        # Iterate through each task file
        for task_file in tqdm(task_files, desc=f"Processing tasks in {model_dir}", leave=False):
            # Read in the task CSV
            task_path = os.path.join(model_path, task_file)
            df = pd.read_csv(task_path)

            try:
                # Ensure the scores are floats
                df['rougel_task_score'] = df['rougel_task_score'].astype(float)
            except Exception as e:
                print(f"Error processing {task_path}: {e}. Skipping...")
                continue

            # Extract the task name to identify the category
            task_name = "_".join(Path(task_file).stem.split("_")[1:])
            task_file_path = f"./data/tasks/{task_name}.json"

            print(model_dir, task_file, task_file_path)
            task_category = task_to_category.get(task_file_path, "Unknown Category")

            # Identify the base instructions (rows without paraphrase type/group)
            base_rows = df[(df['paraphrase_type'].isna()) & (df['paraphrase_group'].isna())]

            # Identify all paraphrased rows
            paraphrased_rows = df[~((df['paraphrase_type'].isna()) & (df['paraphrase_group'].isna()))]

            # For each base instruction, find the corresponding paraphrases and their best score
            for _, base_row in base_rows.iterrows():
                base_instruction = base_row['instruction']
                base_input = base_row['input']
                base_score = base_row['rougel_task_score']

                # Find matching paraphrased rows with the same original instruction and input
                matching_paraphrases = paraphrased_rows[
                    (paraphrased_rows['original_instruction'] == base_instruction) &
                    (paraphrased_rows['input'] == base_input)
                ]

                # Separate paraphrases with better and worse scores than the base
                better_than_base = matching_paraphrases[matching_paraphrases['rougel_task_score'] >= base_score]
                worse_than_base = matching_paraphrases[matching_paraphrases['rougel_task_score'] <= base_score]

                # Calculate the gain/loss based on the average method
                gain_score = None
                loss_score = None

                if average_method == "min":
                    gain_score = better_than_base['rougel_task_score'].min() if not better_than_base.empty else None
                    loss_score = worse_than_base['rougel_task_score'].max() if not worse_than_base.empty else None
                elif average_method == "max":
                    gain_score = better_than_base['rougel_task_score'].max() if not better_than_base.empty else None
                    loss_score = worse_than_base['rougel_task_score'].min() if not worse_than_base.empty else None
                elif average_method == "mean":
                    gain_score = better_than_base['rougel_task_score'].mean() if not better_than_base.empty else None
                    loss_score = worse_than_base['rougel_task_score'].mean() if not worse_than_base.empty else None
                elif average_method == "median":
                    gain_score = better_than_base['rougel_task_score'].median() if not better_than_base.empty else None
                    loss_score = worse_than_base['rougel_task_score'].median() if not worse_than_base.empty else None

                lexical_diversity = None
                # Calculate lexical diversity measures for the base instruction
                try:
                    # Add lexical diversity measures for each matching paraphrase
                    lexical_diversity = get_lexical_diversity(base_row['lm_output'])
                except Exception:
                    pass

                # Add the gain results to the final results if available
                if gain_score is not None:
                    results.append({
                        "model": model_dir,
                        "task": task_name,
                        "task_category": task_category,
                        "paraphrase_type": better_than_base['paraphrase_type'].iloc[0],
                        "paraphrase_group": better_than_base['paraphrase_group'].iloc[0],
                        "potential_rouge_l_gain": gain_score - base_score,
                        "potential_rouge_l_gain_percent": (gain_score / base_score) * 100 if base_score != 0 else 0,
                        "potential_rouge_l_loss": 0.0,
                        "potential_rouge_l_loss_percent": 0.0,
                        "average_base_performance": base_score,
                        "rttr": lexical_diversity['rttr'] if lexical_diversity else None,
                        "maas": lexical_diversity['maas'] if lexical_diversity else None,
                        "mattr": lexical_diversity['mattr'] if lexical_diversity else None,
                        "mtld": lexical_diversity['mtld'] if lexical_diversity else None,
                    })

                # Add the loss results to the final results if available
                if loss_score is not None:
                    results.append({
                        "model": model_dir,
                        "task": task_name,
                        "task_category": task_category,
                        "paraphrase_type": worse_than_base['paraphrase_type'].iloc[0],
                        "paraphrase_group": worse_than_base['paraphrase_group'].iloc[0],
                        "potential_rouge_l_gain": 0.0,
                        "potential_rouge_l_gain_percent": 0.0,
                        "potential_rouge_l_loss": loss_score - base_score,
                        "potential_rouge_l_loss_percent": (loss_score / base_score) * 100 if base_score != 0 else 0,
                        "average_base_performance": base_score,
                        "rttr": lexical_diversity['rttr'] if lexical_diversity else None,
                        "maas": lexical_diversity['maas'] if lexical_diversity else None,
                        "mattr": lexical_diversity['mattr'] if lexical_diversity else None,
                        "mtld": lexical_diversity['mtld'] if lexical_diversity else None,
                    })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Group by task and calculate averages
    task_avg_df = results_df.groupby(['task', 'task_category']).agg({
        'potential_rouge_l_gain': 'mean',
        'potential_rouge_l_loss': 'mean',
        'potential_rouge_l_gain_percent': 'mean',
        'potential_rouge_l_loss_percent': 'mean',
        'average_base_performance': 'mean'
    }).reset_index()

    # Save the DataFrame to a new CSV
    task_avg_df.to_csv(os.path.join(results_path, f"potential_performance_gains_losses_by_tasks_{average_method}.csv"), index=False)

    # Group by model and calculate averages
    model_avg_df = results_df.groupby('model').agg({
        'potential_rouge_l_gain': 'mean',
        'potential_rouge_l_loss': 'mean',
        'potential_rouge_l_gain_percent': 'mean',
        'potential_rouge_l_loss_percent': 'mean',
        'average_base_performance': 'mean',
        'rttr': 'mean',
        'maas': 'mean',
        'mattr': 'mean',
        'mtld': 'mean'
    }).reset_index()

    # Save the DataFrame to a new CSV
    model_avg_df.to_csv(os.path.join(results_path, f"potential_performance_gains_losses_by_models_{average_method}.csv"), index=False)

    # Print per model averages
    for index, row in model_avg_df.iterrows():
        print(f"Model: {row['model']}, {average_method} Rouge-L Gain: {row['potential_rouge_l_gain']}, {average_method} Base Performance: {row['average_base_performance']}")

    # Group by paraphrase type and group, and calculate averages
    paraphrase_avg_df = results_df.groupby(['paraphrase_type', 'paraphrase_group']).agg({
        'potential_rouge_l_gain': 'mean',
        'potential_rouge_l_loss': 'mean',
        'potential_rouge_l_gain_percent': 'mean',
        'potential_rouge_l_loss_percent': 'mean',
        'average_base_performance': 'mean',
        'rttr': 'mean',
        'maas': 'mean',
        'mattr': 'mean',
        'mtld': 'mean'
    }).reset_index()

    # Save the DataFrame to a new CSV
    paraphrase_avg_df.to_csv(os.path.join(results_path, f"potential_performance_gains_losses_by_paraphrases_{average_method}.csv"), index=False)

    # Print per paraphrase averages for each type and group
    results_df.groupby(['task', 'task_category', 'paraphrase_type', 'paraphrase_group']).agg({
        'potential_rouge_l_gain': 'mean',
        'potential_rouge_l_loss': 'mean',
        'potential_rouge_l_gain_percent': 'mean',
        'potential_rouge_l_loss_percent': 'mean',
        'average_base_performance': 'mean',
        'rttr': 'mean',
        'maas': 'mean',
        'mattr': 'mean',
        'mtld': 'mean'
    }).reset_index()

    # Save the DataFrame to a new CSV
    results_df.to_csv(os.path.join(results_path, f"potential_performance_gains_losses_by_paraphrases_tasks_{average_method}.csv"), index=False)

    # Print the overall statistics for gains and losses
    overall_average_gain = results_df['potential_rouge_l_gain'].mean()
    overall_average_loss = results_df['potential_rouge_l_loss'].mean()
    overall_average_base = results_df['average_base_performance'].mean()
    overall_average_gain_percent = results_df['potential_rouge_l_gain_percent'].mean()
    overall_average_loss_percent = results_df['potential_rouge_l_loss_percent'].mean()
    overall_average_rttr = results_df['rttr'].mean()
    overall_average_maas = results_df['maas'].mean()
    overall_average_mattr = results_df['mattr'].mean()
    overall_average_mtld = results_df['mtld'].mean()

    print(f"Overall {average_method} Rouge-L improvement: {overall_average_gain}")
    print(f"Overall {average_method} Rouge-L improvement percent: {overall_average_gain_percent}%")
    print(f"Overall {average_method} Rouge-L loss: {overall_average_loss}")
    print(f"Overall {average_method} Rouge-L loss percent: {overall_average_loss_percent}%")
    print(f"Overall {average_method} base performance: {overall_average_base}")
    print(f"Overall {average_method} RTTR: {overall_average_rttr}")
    print(f"Overall {average_method} Maas: {overall_average_maas}")
    print(f"Overall {average_method} MATTR: {overall_average_mattr}")
    print(f"Overall {average_method} MTLD: {overall_average_mtld}")

def visualize_how_gain_loss_is_computed(results_path, model_name, task_name, output_file):
    # Read the task CSV file
    task_path = os.path.join(results_path, model_name, f"{task_name}.csv")
    df = pd.read_csv(task_path)

    # Convert the ROUGE-L scores to floats for sorting
    df['rougel_task_score'] = df['rougel_task_score'].astype(float)

    # Identify all base instructions (without paraphrase group/type)
    base_rows = df[(df['paraphrase_type'].isna()) & (df['paraphrase_group'].isna())]

    # Initialize variable to hold a suitable base instruction
    selected_base_row = None

    # Iterate through each base instruction to find a suitable example
    for _, base_row in base_rows.iterrows():
        base_instruction = base_row['instruction']
        base_input = base_row['input']
        base_score = base_row['rougel_task_score']

        # Find matching paraphrased rows
        matching_paraphrases = df[
            (df['original_instruction'] == base_instruction) &
            (df['input'] == base_input) &
            (~((df['paraphrase_type'].isna()) & (df['paraphrase_group'].isna())))
        ]

        # Filter paraphrases better than the base score
        better_than_base = matching_paraphrases[matching_paraphrases['rougel_task_score'] > base_score]

        # Check if this base instruction has around 10 to 15 better paraphrases
        if 7 < len(better_than_base) < 16 and base_score > 0.3:
            selected_base_row = base_row
            sorted_rows = pd.concat([pd.DataFrame([base_row]), matching_paraphrases]).sort_values(by='rougel_task_score', ascending=False)
            break

    # If no suitable base instruction is found, return without writing the file
    if selected_base_row is None:
        print(f"No suitable example found for task: {task_name}")
        return

    # Get the base score and the best score
    base_score = selected_base_row['rougel_task_score']
    best_score = sorted_rows.iloc[0]['rougel_task_score']

    # Print the true output using the "output" column of the dataset
    true_output = selected_base_row['output']
    print(f"True output: {true_output}")

    # Abbreviate and clean up paraphrase groups and types
    def clean_group_type(group, type_):
        group_mapping = {
            "Syntax-based changes": "Syntax",
            "Morphology-based changes": "Morphology",
            "Lexicon-based changes": "Lexicon",
            "Lexico-syntactic based changes": "Lexico-Syntax",
            "Discourse-based changes": "Discourse",
            "Extremes": "Extremes",
            "Others": "Others"
        }
        type_ = type_.replace("changes", "").replace("Substitution", "Sub.").replace("Opposite", "Opp.").strip()
        group = group_mapping.get(group, group)
        return group, type_.title()

    # Write LaTeX table
    with open(output_file, 'w') as f:
        f.write("\\begin{table*}\n")
        f.write("    \\begin{tabular}{l l l r}\n")
        f.write("    \\toprule\n")
        f.write("    Paraphrase Group & Paraphrase Type & Model Output & ROUGE-L \\\\\n")
        f.write("    \\midrule\n")

        # Iterate through sorted rows to write table content
        for _, row in sorted_rows.iterrows():
            highlight = ""
            if row['rougel_task_score'] == best_score:
                highlight = "\\hlgold"
            elif row['rougel_task_score'] > base_score:
                highlight = "\\hlgreen"

            paraphrase_group = row['paraphrase_group'] if pd.notna(row['paraphrase_group']) else "Original"
            paraphrase_type = row['paraphrase_type'] if pd.notna(row['paraphrase_type']) else "Original"

            # Apply the cleaning function to groups and types
            paraphrase_group, paraphrase_type = clean_group_type(paraphrase_group, paraphrase_type)

            # Properly escape underscores in the output
            model_output = row['lm_output'].replace("_", "\\_")

            # Format the score to two decimal places
            score_str = f"{row['rougel_task_score']:.2f}"

            f.write(f"    {paraphrase_group} & {paraphrase_type} & {model_output} & {highlight}{{{score_str}}} \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\n")
        f.write(f"    \\caption{{This table demonstrates the potential gain in performance for the task {task_name}. \\hlgold{{Max}}: The highest-scoring paraphrase is highlighted in gold. \\hlgreen{{Median}}: All scores above the original are highlighted in green, demonstrating improvement over the base prompt.}}\n")
        f.write("    \\label{ap:potential_gain_explained}\n")
        f.write("\\end{table*}\n")

def compute_input_output_tokens():
    with open('data/sample_tasks.txt', 'r') as file:
        for line in file:
            file_path = line.strip()
            main_dataset_inputs, main_dataset_outputs, _, _, _, _, _, _= load_dataset(file_path)
            num_input_tokens = sum(len(inputs.page_content.split()) for inputs in main_dataset_inputs) / len(main_dataset_inputs)
            num_output_tokens = sum(len(outputs.page_content.split()) for outputs in main_dataset_outputs) / len(main_dataset_outputs)

            print(f"Dataset: {file_path}")
            print(f"Avg. number of input tokens: {num_input_tokens:.1f}")
            print(f"Avg. number of output tokens: {num_output_tokens:.1f}")

def compute_gain_loss_for_paraphrase_groups_and_task_categories(data_path):
    # Load data from the CSV
    df = pd.read_csv(os.path.join(data_path, "results", "potential_performance_gains_losses_by_paraphrases_tasks_mean.csv"))

    # Group data by model and task category
    grouped = df.groupby(['model', 'task_category', 'paraphrase_group'])

    # Initialize an empty dictionary to store results per model
    results = {}

    def format_gain_loss(gain, loss, base_performance):
        gain_percentage = (gain / base_performance * 100) if base_performance != 0 else 0
        loss_percentage = (loss / base_performance * 100) if base_performance != 0 else 0
        if abs(loss_percentage) >= gain_percentage and loss_percentage < 0.0:
            if -3.0 <= loss_percentage <= 3.0:
                return f"-{abs(loss_percentage):.1f}\\%"
            else:
                return f"\\textcolor{{red}}{{-{abs(loss_percentage):.1f}\\% $\\downarrow$}}"
        elif gain_percentage > abs(loss_percentage) and gain_percentage > 0.0:
            if -3.0 <= gain_percentage <= 3.0:
                return f"+{abs(gain_percentage):.1f}\\%"
            else:
                return f"\\textcolor{{forestgreen}}{{+{gain_percentage:.1f}\\% $\\uparrow$}}"
        else:
            return "0.0\\%"

    # Iterate over grouped data to accumulate gains and losses
    for (model, task_category, paraphrase_group), group_data in grouped:
        gain = group_data['potential_rouge_l_gain'].mean()
        loss = group_data['potential_rouge_l_loss'].mean()
        base_performance = group_data['average_base_performance'].mean()

        if model not in results:
            results[model] = {}
        if task_category not in results[model]:
            results[model][task_category] = {}

        results[model][task_category][paraphrase_group] = format_gain_loss(gain, loss, base_performance)

    # Generate LaTeX table per model
    latex_tables = {}

    for model, task_data in results.items():
        latex = "\\begin{tabular}{lcccccc}\n\\toprule\n"
        latex += "\\text{Task Category} & \\text{Morphology} & \\text{Syntax} & \\text{Lexicon} & \\text{Lexico-Syntax} & \\text{Discourse} & \\text{Others} \\\\\n\\midrule\n"

        for task_category, paraphrase_data in task_data.items():
            morphology = paraphrase_data.get('Morphology-based changes', "0.0\\%")
            syntax = paraphrase_data.get('Syntax-based changes', "0.0\\%")
            lexicon = paraphrase_data.get('Lexicon-based changes', "0.0\\%")
            lexico_syntax = paraphrase_data.get('Lexico-syntactic based changes', "0.0\\%")
            discourse = paraphrase_data.get('Discourse-based changes', "0.0\\%")
            others = paraphrase_data.get('Others', "0.0\\%")

            latex += f"\\text{{{task_category}}} & {morphology} & {syntax} & {lexicon} & {lexico_syntax} & {discourse} & {others} \\\\\n"

        latex += "\\bottomrule\n\\end{tabular}\n"

        latex_tables[model] = latex

    # Print or save the LaTeX tables as needed
    for model, latex in latex_tables.items():
        with open(os.path.join(data_path, "tables", f"{model}_groups_by_tasks.tex"), "w") as f:
            f.write(latex)


if __name__ == '__main__':
    for average_method in ["max", "median", "mean"]:
        get_potential_performance_gain("outputs/results/", task_to_category_path="data/sample_tasks_to_categories.json", average_method=average_method)
    compute_gain_loss_for_paraphrase_groups_and_task_categories("outputs/")
    # visualize_how_gain_loss_is_computed("outputs/results", "llama-3-8b-instruct", "meta-llama-Meta-Llama-3-8B-Instruct_task589_amazonfood_summary_text_generation", "outputs/tables/potential_gain_explained.tex")
