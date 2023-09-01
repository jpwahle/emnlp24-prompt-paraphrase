import os
import json
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def query_dataset_batch(queries, dataset_with_index, k=1):
    scores, retrieved_examples = dataset_with_index.get_nearest_examples_batch("text", queries, k=k, request_timeout=50000)
    return scores, retrieved_examples

def calculate_rouge_l_and_cosine_similarity_batch(dataset_with_index, prompts):
    # Query the dataset in batch
    bm25_scores, top_examples = query_dataset_batch(prompts, dataset_with_index)

    # Initialize ROUGE scorer
    rouge = Rouge()

    # Calculate ROUGE-L scores
    rouge_scores = []
    for prompt, example in zip(prompts, top_examples):
        score = rouge.get_scores(prompt, example['text'][0])[0]["rouge-l"]["f"]
        rouge_scores.append(score)

    return rouge_scores, bm25_scores

def compute_closeness_to_training_data(dataset, results_path, task_to_category_path, limit_examples=50):

    # Load the task-to-category mapping
    with open(task_to_category_path, 'r') as file:
        task_to_category = json.load(file)

    # List of subdirectories (one per model)
    model_dir = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d)) and "llama-3-70b" in d][0]

    # Initialize a list to hold results
    results = []

    # Iterate through each model directory
    model_path = os.path.join(results_path, model_dir)
    task_files = [f for f in os.listdir(model_path) if f.endswith(".csv")]

    # Iterate through each task file
    for task_file in tqdm(task_files[:100], desc=f"Processing tasks in {model_dir}", leave=False):
        # Read in the task CSV
        task_path = os.path.join(model_path, task_file)
        df = pd.read_csv(task_path)
        df = df.sample(n=limit_examples)

        try:
            # Ensure the scores are floats
            df['rougel_task_score'] = df['rougel_task_score'].astype(float)
        except Exception as e:
            print(f"Error processing {task_path}: {e}. Skipping...")
            continue

        # Extract the task name to identify the category
        task_name = "_".join(Path(task_file).stem.split("_")[1:])
        task_file_path = f"./data/tasks/{task_name}.json"

        task_category = task_to_category.get(task_file_path, "Unknown Category")

        # Prepare instructions and original instructions
        instructions = []
        original_instructions = []
        for _, row in df.iterrows():
            if pd.isna(row['instruction']) or pd.isna(row['original_instruction']):
                continue
            instructions.append(row['instruction'])
            original_instructions.append(row['original_instruction'] if 'original_instruction' in row else row['instruction'])

        # Get closeness scores for all instructions and original instructions in batch
        paraphrased_rouge_scores, paraphrased_bm25_scores = calculate_rouge_l_and_cosine_similarity_batch(dataset, instructions)
        original_rouge_scores, original_bm25_scores = calculate_rouge_l_and_cosine_similarity_batch(dataset, original_instructions)

        valid_index = 0
        for _, row in df.iterrows():
            if pd.isna(row['instruction']) or pd.isna(row['original_instruction']):
                continue

            results.append({
                "model": model_dir,
                "task": task_name,
                "task_category": task_category,
                "paraphrase_type": row.get('paraphrase_type', None),
                "paraphrase_group": row.get('paraphrase_group', None),
                "rougel_task_score": row['rougel_task_score'],
                "paraphrased_bm25": paraphrased_bm25_scores[valid_index][0],
                "paraphrased_rouge": paraphrased_rouge_scores[valid_index],
                "original_bm25": original_bm25_scores[valid_index][0],
                "original_rouge": original_rouge_scores[valid_index]
            })

            valid_index += 1

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Compute averages per task
    task_avg_df = results_df.groupby(['task', 'task_category']).agg({
        'rougel_task_score': 'mean',
        'paraphrased_bm25': 'mean',
        'paraphrased_rouge': 'mean',
        'original_bm25': 'mean',
        'original_rouge': 'mean'
    }).reset_index()
    task_avg_df.to_csv(os.path.join(results_path, "average_closeness_per_task.csv"), index=False)

    # Compute averages per paraphrase type and group
    paraphrase_avg_df = results_df.groupby(['paraphrase_type', 'paraphrase_group']).agg({
        'rougel_task_score': 'mean',
        'paraphrased_bm25': 'mean',
        'paraphrased_rouge': 'mean',
        'original_bm25': 'mean',
        'original_rouge': 'mean'
    }).reset_index()
    paraphrase_avg_df.to_csv(os.path.join(results_path, "average_closeness_per_paraphrase.csv"), index=False)

    # Compute averages per model
    model_avg_df = results_df.groupby('model').agg({
        'rougel_task_score': 'mean',
        'paraphrased_bm25': 'mean',
        'paraphrased_rouge': 'mean',
        'original_bm25': 'mean',
        'original_rouge': 'mean'
    }).reset_index()
    model_avg_df.to_csv(os.path.join(results_path, "average_closeness_per_model.csv"), index=False)

    # Compute per types and tasks
    task_avg_df = results_df.groupby(['task', 'task_category', 'paraphrase_type', 'paraphrase_group']).agg({
        'rougel_task_score': 'mean',
        'paraphrased_bm25': 'mean',
        'paraphrased_rouge': 'mean',
        'original_bm25': 'mean',
        'original_rouge': 'mean'
    }).reset_index()

    # Store the individual results
    results_df.to_csv(os.path.join(results_path, "individual_closeness.csv"), index=False)

    return task_avg_df, paraphrase_avg_df, model_avg_df

# Load the dataset
def plot_closeness(file_path):
    data = pd.read_csv(file_path)

    # Prepare data for analysis
    data['rouge_difference'] = abs(data['paraphrased_rouge'] - data['original_rouge'])

    # Filtering the data for task scores <0.02 and >0.98
    filtered_data = data[(data['rougel_task_score'] >= 0.02) & (data['rougel_task_score'] <= 0.98)]

    # Joint distribution plot for ROUGE differences and task scores with hex bins and colored marginal histograms
    plt.figure(figsize=(10, 8))

    # Create jointplot with hex bins
    joint_plot = sns.jointplot(x='rouge_difference', y='rougel_task_score', data=filtered_data, kind='hex', marginal_kws=dict(bins=30, fill=True), cmap='RdYlBu_r')

    # Customize the marginal histograms
    joint_plot.ax_marg_x.hist(filtered_data['rouge_difference'], bins=30, color='#5D8DAF', alpha=0.6)
    joint_plot.ax_marg_y.hist(filtered_data['rougel_task_score'], bins=30, orientation='horizontal', color='#5D8DAF', alpha=0.6)

    # Make x and y labels bigger
    joint_plot.ax_joint.set_xlabel('ROUGE-L Between Prompt and Train. Example (Paraphrased - Original)', fontsize=14)
    joint_plot.ax_joint.set_ylabel('ROUGE-L Task Score', fontsize=14)

    # Make xticks and yticks bigger
    joint_plot.ax_joint.tick_params(axis='x', labelsize=15)
    joint_plot.ax_joint.tick_params(axis='y', labelsize=15)

    # Set titles and labels
    joint_plot.set_axis_labels(r'$\Delta_{train}$', 'ROUGE-L Task Score')

    # Store as PDF
    plt.savefig("outputs/figures/rouge_difference_vs_task_score.pdf", format='pdf')

if __name__ == '__main__':
    # dataset_name = 'HuggingFaceFW/fineweb'
    # sample_name = 'sample-350BT'
    # split = 'train'

    # fineweb = load_dataset(dataset_name, sample_name, split=split, num_proc=12, revision="042ac03070484d97ab32e6899e1c2b571b2e9c38")
    # fineweb.load_elasticsearch_index("text", host="localhost", port="9200", es_index_name=f"hf_{dataset_name.replace('/', '_')}_{sample_name}_{split}_text".lower())
    # compute_closeness_to_training_data(fineweb, "outputs/results/", "data/sample_tasks_to_categories.json")

    plot_closeness("outputs/results/individual_closeness.csv")
