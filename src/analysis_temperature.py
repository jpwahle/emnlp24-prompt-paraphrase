import json
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_community.document_loaders.json_loader import JSONLoader
from openai import OpenAI
from rouge import Rouge
from scipy.interpolate import griddata
from tqdm import tqdm

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/results").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARN
)  # Adjust to logging.INFO to see more information

openai_client = OpenAI()

CHATGPT_MODEL_NAME = "gpt-3.5-turbo-16k"
PARAPHRSE_TYPE_MODEL_NAME = "ft:gpt-3.5-turbo-0613:personal::7xbU0xQ2"
GROUPED_TYPES = {
    "Morphology-based changes": [
        "Inflectional changes",
        "Modal verb changes",
        "Derivational changes",
    ],
    "Lexicon-based changes": [
        "Spelling changes",
        "Change of format",
        "Same Polarity Substitution (contextual)",
        "Same Polarity Substitution (habitual)",
        "Same Polarity Substitution (named ent.)",
    ],
    "Lexico-syntactic based changes": [
        "Converse substitution",
        "Opposite polarity substitution (contextual)",
        "Opposite polarity substitution (habitual)",
        "Synthetic/analytic substitution",
    ],
    "Syntax-based changes": [
        "Coordination changes",
        "Diathesis alternation",
        "Ellipsis",
        "Negation switching",
        "Subordination and nesting changes",
    ],
    "Discourse-based changes": [
        "Direct/indirect style alternations",
        "Punctuation changes",
        "Syntax/discourse structure changes",
    ],
    "Extremes": ["Entailment", "Identity", "Non-paraphrase"],
    "Others": ["Addition/Deletion", "Change of order", "Semantic-based"],
}


def create_paraphrase_prompt(sentence, paraphrase_type):
    return [
            {
                "role": "user",
                "content": (
                    "Given the following sentence, generate a paraphrase with"
                    f" the following types. Sentence: {sentence}. Paraphrase"
                    f" Types: {paraphrase_type}"
                ),
            }
        ]

def load_dataset(file_path: str):

    truncate_by = 4000

    if "cuad" in file_path.lower():
        truncate_by = 3000

    logging.info("Executing load_dataset function")
    main_dataset_input_loader = JSONLoader(
        file_path=file_path, jq_schema=".Instances[].input"
    )
    main_dataset_inputs = main_dataset_input_loader.load()

    # For CUAD dataset: Check if any of the inputs are longer than 7000 tokens (with 5 characters per token) and shorten them
    for i, inputs in enumerate(main_dataset_inputs):
        if len(inputs.page_content) > truncate_by:
            main_dataset_inputs[i].page_content = inputs.page_content[:truncate_by]

    main_dataset_output_loader = JSONLoader(
        file_path=file_path, jq_schema=".Instances[].output[]"
    )
    main_dataset_outputs = main_dataset_output_loader.load()

    positive_example_loader = JSONLoader(
        file_path=file_path,
        jq_schema='."Positive Examples"[] | "Input: \(.input)\nOutput: \(.output)"', # type: ignore
    )
    positive_examples = positive_example_loader.load()

    negative_example_loader = JSONLoader(
        file_path=file_path,
        jq_schema='."Negative Examples"[] | "Input: \(.input)\nOutput: \(.output)"', # type: ignore
    )
    negative_examples = negative_example_loader.load()

    # For CUAD dataset: Check if positive and negative examples are too long (more than 1000 characters each)
    for i, example in enumerate(positive_examples):
        if len(example.page_content) > truncate_by:
            positive_examples[i].page_content = example.page_content[:truncate_by]
            logging.warning("A positive example was truncated due to excessive length.")

    for i, example in enumerate(negative_examples):
        if len(example.page_content) > truncate_by:
            negative_examples[i].page_content = example.page_content[:truncate_by]
            logging.warning("A negative example was truncated due to excessive length.")

    reasoning_loader = JSONLoader(
        file_path=file_path,
        jq_schema='."Reasoning"[]',
    )
    reasoning = reasoning_loader.load()

    instruction = json.load(open(file_path, "r"))["Definition"][0]
    domains = json.load(open(file_path, "r"))["Categories"]  # Extract all domains
    task_name = os.path.basename(file_path).split(".")[
        0
    ]  # Extract task name from file path

    return (
        main_dataset_inputs,
        main_dataset_outputs,
        positive_examples,
        negative_examples,
        reasoning,
        instruction,
        domains,
        task_name,
    )

def compute_rouge(hypothesis, reference):
    """
    Computes ROUGE-L score between reference and hypothesis.

    :param reference: The ground truth sentence.
    :param hypothesis: The generated sentence by the language model.
    :return: ROUGE-L score.
    """
    rouge = Rouge()
    if reference and hypothesis:
        if len(reference) != 0 and len(hypothesis) != 0:
            scores = rouge.get_scores(hypothesis, reference)
            if scores and isinstance(scores, list) and scores[0] and "rouge-l" in scores[0]:
                return scores[0]["rouge-l"]["f"]  # Using F1-score of ROUGE-L
    else:
        return 0

def paraphrase_prompt(prompt, paraphrase_type):
    model_prompt = create_paraphrase_prompt(prompt, paraphrase_type)
    new_prompt = None
    try:
        response = (
            openai_client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0613:personal::7xbU0xQ2",
                messages=model_prompt, # type: ignore
            )
        )
        new_prompt = response.choices[0].message.content

    except Exception as e:
        if hasattr(e, 'headers'):
            retry_after = e.headers.get('Retry-After', 4) # type: ignore
            logging.warning(
                f"Error: OpenAI request failed. Retrying after {retry_after} seconds..."
            )
            time.sleep(retry_after)
            return paraphrase_prompt(prompt, paraphrase_type)
        else:
            logging.error(f"Error: {e}")
            time.sleep(4)
            return paraphrase_prompt(prompt, paraphrase_type)

    return new_prompt

def run_temperature_analysis(task_paths, limit_examples=10, temperatures=np.linspace(0, 1, 11), paraphrase_groups=["Morphology-based changes", "Lexicon-based changes"] ):
    results = []
    for task_path in tqdm(task_paths):
        main_dataset_inputs, main_dataset_outputs, positive_examples, negative_examples, reasoning, instruction, domains, task_name = load_dataset(task_path)
        types_to_consider = [ptype for key, types in GROUPED_TYPES.items() for ptype in types if key in paraphrase_groups]
        for paraphrase_type in tqdm(types_to_consider):
            paraphrase_instruction = create_paraphrase_prompt(instruction, paraphrase_type)
            for temperature_orig in tqdm(temperatures[::2]):
                for temperature_para in tqdm(temperatures[1:][::2]):
                    for main_dataset_input, main_dataset_output in zip(main_dataset_inputs[:limit_examples], main_dataset_outputs[:limit_examples]):
                        prompt_original=f"Instruction: {instruction}\n\nPositive examples.\n{positive_examples}\n\nNegative examples.\n{negative_examples}\n\nNow, predict the output for the following example.\nInput: {main_dataset_input.page_content}\nOutput:"
                        system_prompt = "You are a helpful assistant in solving various tasks. You should only output one answer to the task, nothing more, no explanations, and nothing around. Just read the instruction carefully, understand the positive and negative examples provided, and generate one single answer in the same way as the example's output."
                        messages_original = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_original}
                        ]
                        response_orig = (
                            openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages_original, # type: ignore
                                temperature=temperature_orig,
                            )
                        )
                        prompt_para=f"Instruction: {paraphrase_instruction}\n\nPositive examples.\n{positive_examples}\n\nNegative examples.\n{negative_examples}\n\nNow, predict the output for the following example.\nInput: {main_dataset_input.page_content}\nOutput:"
                        system_prompt = "You are a helpful assistant in solving various tasks. You should only output one answer to the task, nothing more, no explanations, and nothing around. Just read the instruction carefully, understand the positive and negative examples provided, and generate one single answer in the same way as the example's output."
                        messages_para= [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_para}
                        ]
                        response_para = (
                            openai_client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages_para, # type: ignore
                                temperature=temperature_para,
                            )
                        )

                        # Compute ROUGE-L score between original and paraphrased responses
                        rouge_score_original = compute_rouge(response_orig.choices[0].message.content, main_dataset_output.page_content)
                        rouge_score_paraphrased = compute_rouge(response_para.choices[0].message.content, main_dataset_output.page_content)

                        paraphrase_group = None
                        if paraphrase_type:
                            for group, types in GROUPED_TYPES.items():
                                if paraphrase_type in types:
                                    paraphrase_group = group
                                    break

                        # Save the results
                        res = {
                            "task_name": task_name,
                            "temperature_orig": temperature_orig,
                            "temperature_para": temperature_para,
                            "paraphrase_type": paraphrase_type,
                            "paraphrase_group": paraphrase_group,
                            "original_response": response_orig.choices[0].message.content,
                            "paraphrased_response": response_para.choices[0].message.content,
                            "rouge_score_original": rouge_score_original,
                            "rouge_score_paraphrased": rouge_score_paraphrased,
                        }

                        results.append(res)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/results/temperature_analysis_results.csv', index=False)


# Function to create a contour plot
def create_contour(ax, x, y, z, title, levels, xlabel='', ylabel='', title_pos="bottom"):
    xi, yi = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    contour = ax.contourf(xi, yi, zi, levels=levels, cmap='RdYlBu_r')
    ax.contour(xi, yi, zi, levels=levels, colors='k')
    if title_pos == "top":
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    elif title_pos == "bottom":
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20, loc='center', y=-0.55)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    return contour

# Prepare the data for plotting
def prepare_data_for_plotting(data, task_name, paraphrase_group):
    subset = data[(data['task_name'] == task_name) & (data['paraphrase_group'] == paraphrase_group)]
    x = subset['temperature_orig'].values
    y = subset['temperature_para'].values
    z = subset['rouge_score_paraphrased'].values - subset['rouge_score_original'].values
    return x, y, z

def plot_data(data, tasks, paraphrase_groups):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # Adjust figure size for 1x4 grid

    # Define the levels for the contour plots to ensure consistency
    levels = np.linspace(-1, 1, 20)

    # Generate plots for each combination of task and paraphrase group
    idx = 0
    for task in tasks:
        for paraphrase_group in paraphrase_groups:
            if idx < 4:  # Ensure we do not exceed the 1x4 grid
                x, y, z = prepare_data_for_plotting(data, task, paraphrase_group)
                task_name = task.replace("task1659_title_generation", "Title Generation").replace("task645_summarization", "Summarization")
                paraphrase_group_name = paraphrase_group.replace("Morphology-based changes", "Morphology").replace("Lexicon-based changes", "Lexicon").replace("Lexico-syntactic based changes", "Lexico-syntactic").replace("Syntax-based changes", "Syntax").replace("Discourse-based changes", "Discourse")
                title = f'({chr(97 + idx)}) {paraphrase_group_name} x {task_name}'
                create_contour(axs[idx], x, y, z, title, levels=levels,
                               xlabel='Temperature Para.',
                               ylabel='Temperature Orig.' if idx == 0 else '',
                               title_pos="bottom")
                idx += 1

    # Create a single colorbar
    cbar = fig.colorbar(axs[0].collections[0], ax=axs, orientation='vertical', fraction=0.02, pad=0, aspect=35)
    cbar.set_label(r'$\Delta_{temp}$')
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.yaxis.label.set_rotation(270)
    cbar.ax.yaxis.set_label_coords(10, 0.5)
    cbar.set_ticks(np.arange(-1, 1.1, 0.2))
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig('outputs/figures/temperature.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    # Run model experiments
    # run_temperature_analysis(["data/tasks/task1659_title_generation.json", "data/tasks/task645_summarization.json"])

    # Load the output CSV file
    file_path = 'outputs/results/temperature_analysis_results.csv'
    data = pd.read_csv(file_path)

    # Extracting unique tasks and paraphrase groups
    tasks = data['task_name'].unique()
    paraphrase_groups = data['paraphrase_group'].unique()

    plot_data(data, tasks, paraphrase_groups)
