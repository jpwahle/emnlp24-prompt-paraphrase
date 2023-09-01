import spacy
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from paraphrase_metrics import metrics as pm

tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

def group_and_compute_dependence(df):

    # Cast the instruction and original_instruction columns to string
    df["instruction"] = df["instruction"].astype(str)
    df["original_instruction"] = df["original_instruction"].astype(str)

    df["instruction_token_count"] = df["instruction"].apply(lambda x: len(x.split()))
    df["instruction_wpd"] = df.progress_apply(lambda x: pm.wpd(nlp(x.instruction), nlp(x.original_instruction)), axis=1)
    df["instruction_ld"] = df.progress_apply(lambda x: pm.ld(nlp(x.instruction), nlp(x.original_instruction)), axis=1)

    # Grouping the DataFrame
    grouped_df = df.groupby(df.index // 27)

    # Compute the mean of instruction_token_count and rougel_task_score for each group
    mean_token_count = grouped_df["instruction_token_count"].mean()
    mean_wpd = grouped_df["instruction_wpd"].mean()
    mean_ld = grouped_df["instruction_ld"].mean()
    mean_rougel_score = grouped_df["rougel_task_score"].mean()

    # Function to compute deviation for each group
    def compute_deviation(group, mean_token_count, mean_wpd, mean_ld, mean_rougel_score):
        group["deviation_token_count"] = abs(
            group["instruction_token_count"] - mean_token_count[group.name]
        )
        group["deviation_rougel_task_score"] = abs(
            group["rougel_task_score"] - mean_rougel_score[group.name]
        )
        group["deviation_wpd"] = abs(
            group["instruction_wpd"] - mean_wpd[group.name]
        )
        group["deviation_ld"] = abs(
            group["instruction_ld"] - mean_ld[group.name]
        )
        return group

    # Apply the function to each group and combine the results
    result_df = grouped_df.apply(compute_deviation, mean_token_count=mean_token_count, mean_wpd=mean_wpd, mean_ld=mean_ld, mean_rougel_score=mean_rougel_score)

    return result_df

def compute_correlation_and_p_value_per_task(df):
    # Initialize a dictionary to store the correlation and p-value for each metric
    task_correlation_p_value = {}

    # Metrics to correlate with deviation_rougel_task_score
    metrics = ["deviation_token_count", "deviation_wpd", "deviation_ld"]

    for metric in metrics:
        # Extract the series for the current metric and deviation_rougel_task_score
        deviation_metric = df[metric]
        deviation_rougel_task_score = df["deviation_rougel_task_score"]

        # Compute the Pearson correlation and p-value
        correlation, p_value = pearsonr(
            deviation_metric, deviation_rougel_task_score
        )

        # Store the correlation and p-value for the current metric
        task_correlation_p_value[metric] = (correlation, p_value)

    return task_correlation_p_value


def average_results_from_tex(file_path):
    # Read the LaTeX table file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the table data
    table_data = []
    inside_table = False
    for line in lines:
        if '\\midrule' in line:
            inside_table = True
            continue
        if '\\bottomrule' in line:
            break
        if inside_table:
            table_data.append(line.strip())

    # Parse the table data
    parsed_data = []
    for row in table_data:
        if '&' in row:
            parsed_data.append([item.strip() for item in row.split('&')])

    # Create a DataFrame
    columns = ["Model", "Task", "Metric", "Correlation", "P-value"]
    df = pd.DataFrame(parsed_data, columns=columns)

    # Convert numeric columns to float
    df["Correlation"] = df["Correlation"].str.replace('*', '').astype(float)
    df["P-value"] = df["P-value"].str.replace('\\', '').astype(float)

    # Map in the metric column to the full metric name deviation_token_count: Tok, deviation_wpd: Pos, deviation_ld: Lex
    df["Metric"] = df["Metric"].map({
        "deviation_token_count": "Tok",
        "deviation_wpd": "Pos",
        "deviation_ld": "Lex"
    })

    # Pivot the table to have Lex, Pos, Tok as columns
    pivot_df = df.pivot_table(index="Task", columns="Metric", values=["Correlation", "P-value"], aggfunc="mean")
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

    # Calculate the average P-value across metrics for each task
    pivot_df['Avg_P-value'] = pivot_df[['P-value_Tok', 'P-value_Pos', 'P-value_Lex']].mean(axis=1)

    # Output the result
    output_file = 'outputs/tables/token_performance_correlation_table_avg_pivot.tex'
    with open(output_file, 'w') as f:
        f.write("\\begin{table}\n")
        f.write("\\caption{Averaged Pearson correlations between deviation metrics and downstream task performance}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Task & Corr. Tok & Corr. Pos & Corr. Lex & Avg. P-value \\\\\n")
        f.write("\\midrule\n")
        for task, row in pivot_df.iterrows():
            f.write(f"{task} & {row['Correlation_Tok']:.2f} & {row['Correlation_Pos']:.2f} & {row['Correlation_Lex']:.2f} & {row['Avg_P-value']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Averaged table written to {output_file}")


if __name__ == "__main__":
    # Define paths
    results_path = "outputs/results"
    task_to_category_path="data/sample_tasks_to_categories.json"
    model_dirs = os.listdir(results_path)
    # For now only process the llama-3-70b-instruct model
    model_dirs = ["llama-3-70b-instruct"]
    # model_dirs = [d for d in model_dirs if os.path.isdir(os.path.join(results_path, d))]

    # Initialize a dictionary to store results for all models
    all_task_correlation_p_value = {}

    # Iterate through each model directory
    for model_dir in tqdm(model_dirs, desc="Processing models"):
        model_path = os.path.join(results_path, model_dir)
        task_files = [f for f in os.listdir(model_path) if f.endswith(".csv")]

        # Initialize a dictionary to store results for each task in the current model
        model_task_correlation_p_value = {}

        # Iterate through each task file
        for task_file in tqdm(task_files, desc=f"Processing tasks in {model_dir}", leave=False):
            # Read in the task CSV
            task_path = os.path.join(model_path, task_file)
            df = pd.read_csv(task_path)

            # Filter the df
            grouped_df = group_and_compute_dependence(df)

            # Compute correlation and p value for the current task
            task_correlation_p_value = compute_correlation_and_p_value_per_task(grouped_df)

            # Store the results for the current task
            model_task_correlation_p_value[task_file] = task_correlation_p_value

        # Store the results for the current model
        all_task_correlation_p_value[model_dir] = model_task_correlation_p_value

    # Print the correlation and p value for each task
    for model_dir, tasks in all_task_correlation_p_value.items():
        print(f"Model: {model_dir}")
        for task, metrics in tasks.items():
            print(f"\tTask: {task}")
            for metric, (correlation, p_value) in metrics.items():
                print(f"\tMetric: {metric}, Correlation: {correlation}, P-value: {p_value}")

    # Load the task-to-category mapping
    with open(task_to_category_path, 'r') as file:
        task_to_category = json.load(file)

    # Create a LaTeX table of the correlations
    latex_table = "\\begin{table}\n\\caption{Pearson correlations between deviation metrics and downstream task performance}\n\\begin{tabular}{ccccc}\n\\toprule\nModel & Task & Metric & Correlation & P-value \\\\\n\\midrule"
    for model_dir, tasks in all_task_correlation_p_value.items():
        for task, metrics in tasks.items():
            # Extract the task name to identify the category
            task_name = "_".join(Path(task).stem.split("_")[1:])
            task_file_path = f"./data/tasks/{task_name}.json"
            task_category = task_to_category.get(task_file_path, "Unknown Category")
            for metric, (correlation, p_value) in metrics.items():
                asterisk = "*" if p_value < 0.05 else ""
                latex_table += (
                    f"\n{model_dir} & {task_category} & {metric} & {round(correlation, 2)}{asterisk} & {round(p_value, 2)} \\\\"
                )
    latex_table += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Also store the table as a CSV
    with open("outputs/results/token_performance_correlation_table.csv", "w") as file:
        file.write("Model,Task,Category,Metric,Correlation,P-value\n")
        for model_dir, tasks in all_task_correlation_p_value.items():
            for task, metrics in tasks.items():
                # Extract the task name to identify the category
                task_name = "_".join(Path(task).stem.split("_")[1:])
                task_file_path = f"./data/tasks/{task_name}.json"
                task_category = task_to_category.get(task_file_path, "Unknown Category")
                for metric, (correlation, p_value) in metrics.items():
                    file.write(
                        f"{model_dir},{task},{task_category},{metric},{round(correlation, 2)},{round(p_value, 2)}\n"
                    )

    # Store the LaTeX table in a file
    os.makedirs("outputs/tables", exist_ok=True)
    with open("outputs/tables/token_performance_correlation_table.tex", "w") as file:
        file.write(latex_table)

    # Create an averaged table
    average_results_from_tex("outputs/tables/token_performance_correlation_table.tex")
