import os
import json
import time
import random
import logging
import asyncio
import requests
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rouge import Rouge
from openai import OpenAI
from tqdm.auto import tqdm
from hftgi_chat import HFTGIChat
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders.json_loader import JSONLoader

# Debug
from langchain.globals import set_debug
set_debug(False)

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


def count_tasks_per_domain_and_language(
    language: str = "English", write_csv: bool = True
):
    logging.info("Executing count_tasks_per_domain_and_language function")
    tasks_per_domain = defaultdict(int)

    for json_file in os.listdir("./data/tasks"):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join("./data/tasks", json_file)
        json_data = json.load(open(json_path, "r"))
        if (
            json_data["Input_language"][0] != language
            or json_data["Output_language"][0] != language
            or json_data["Instruction_language"][0] != language
            or any("toxic" in category.lower() for category in json_data["Categories"])
        ):
            continue
        if len(json_data["Categories"]) > 1:
            logging.warning("More than one category", json_data["Categories"])

        tasks_per_domain[json_data["Categories"][0]] += 1

    if write_csv:
        # Export the tasks per domain in an easy to read csv excel sheet
        with open(f"data/count_tasks_per_domain_{language}.csv", "w") as f:
            for domain, count in tasks_per_domain.items():
                f.write(f"{domain},{count}\n")
    return tasks_per_domain


def get_domains_with_n_tasks(tasks_per_domain, n: int = 10):
    logging.info("Executing get_domains_with_n_tasks function")
    return [
        domain
        for domain, count in tasks_per_domain.items()
        if count >= n and domain != "Misc."
    ]


def get_tasknames_to_domains(task_folder: str = "./data/tasks"):
    logging.info("Executing get_tasknames_to_domains function")
    filename_to_domain = {}
    for json_file in os.listdir(task_folder):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join("./data/tasks", json_file)
        json_data = json.load(open(json_path, "r"))
        if len(json_data["Categories"]) > 1:
            logging.warning("More than one category", json_data["Categories"])
        filename_to_domain[json_file] = json_data["Categories"][0]


def sample_tasks_from_domain(domain: str, n: int):
    logging.info("Executing sample_tasks_from_domain function")
    tasks = []
    for json_file in os.listdir("./data/tasks"):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join("./data/tasks", json_file)
        json_data = json.load(open(json_path, "r"))
        if (
            json_data["Input_language"][0] != "English"
            or json_data["Output_language"][0] != "English"
            or json_data["Instruction_language"][0] != "English"
        ):
            continue
        if len(json_data["Categories"]) > 1:
            logging.warning("More than one category", json_data["Categories"])
        if json_data["Categories"][0] == domain:
            tasks.append(json_path)
    return random.sample(tasks, n)


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


def run_llm(
    llmchain,
    instruction,
    main_dataset_inputs,
    main_dataset_outputs,
    positive_examples,
    negative_examples,
    limit_examples,
    reasoning,
    domains,
    task_name,
    llmid,
    original_instruction=None,
    max_pos_examples=3,
    max_neg_examples=3,
    paraphrase_type=None,
    max_concurrency=50,
    stop=None,
    do_async_batch=False
):
    results = []
    batch_inputs = list(
        zip(
            main_dataset_inputs[: limit_examples or len(main_dataset_inputs)],
            main_dataset_outputs[: limit_examples or len(main_dataset_inputs)],
        )
    )

    def async_call(x, options=None, config=None):
        return asyncio.run(llmchain.abatch(x, options=options, config=config))
    batch_fn = async_call if do_async_batch else llmchain.batch

    batch_outputs = batch_fn(
        [
            dict(
                instruction=instruction,
                positive_examples="\n".join(
                    [
                        example.page_content
                        for example in positive_examples[:max_pos_examples]
                    ]
                ),
                negative_examples="\n".join(
                    [
                        example.page_content
                        for example in negative_examples[:max_neg_examples]
                    ]
                ),
                example=inputs.page_content,
            )
            for inputs, _ in batch_inputs
        ],
        options={"stop": stop},
        config={"max_concurrency": max_concurrency},
    )

    for (inputs, outputs), lm_output in zip(batch_inputs, batch_outputs):
        # Compute the rouge score but whenever the output is empty, we give it a score of 0
        rougel_task_score = (
            compute_rouge(outputs.page_content, lm_output["text"])
            if outputs.page_content and len(outputs.page_content) > 0 and lm_output and lm_output["text"] and len(lm_output["text"]) > 0
            else 0
        )

        rougel_prompt = None
        if original_instruction:
            rougel_prompt = compute_rouge(original_instruction, instruction)

        paraphrase_group = None
        if paraphrase_type:
            for group, types in GROUPED_TYPES.items():
                if paraphrase_type in types:
                    paraphrase_group = group
                    break

        for domain in domains:
            result = {
                "input": inputs.page_content,
                "output": outputs.page_content,
                "lm_output": lm_output['text'] if lm_output else None,
                "rougel_task_score": rougel_task_score,
                "rougel_prompt": rougel_prompt,
                "instruction": instruction,
                "original_instruction": original_instruction,
                "domain": domain,
                "reasoning": reasoning,
                "task_name": task_name,
                "paraphrase_type": paraphrase_type,
                "paraphrase_group": paraphrase_group,
                "model": llmid,
            }
            results.append(result)

    return results


def run_tasks(tasks: list, llmchain, prompt, llmid, max_concurrency, limit_examples = None, stop = None, overwrite_existing_outputs = False, do_async_batch = False):
    logging.info("Executing run_tasks function")
    task_tqdm = tqdm(tasks)
    for task in task_tqdm:
        results = []
        (
            main_dataset_inputs,
            main_dataset_outputs,
            positive_examples,
            negative_examples,
            reasoning,
            original_instruction,
            domains,
            task_name,
        ) = load_dataset(task)

        save_llmid = llmid.replace("/", "-") # Replace / with - in llmid
        output_path = f"outputs/results/{save_llmid}_{task_name}.csv"

        if os.path.exists(output_path) and not overwrite_existing_outputs:
            logging.warn(f"Output file already exists: {output_path} and overwrite_existing_outputs is set to False. Skipping task: {task_name}")
            continue

        task_tqdm.set_description(f"Processing task: {task_name}")

        # Run original instruction once
        result = run_llm(
            llmchain=llmchain,
            instruction=original_instruction,
            main_dataset_inputs=main_dataset_inputs,
            main_dataset_outputs=main_dataset_outputs,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            limit_examples=limit_examples,
            domains=domains,
            reasoning=reasoning,
            task_name=task_name,
            llmid=llmid,
            max_concurrency=max_concurrency,
            stop=stop,
            do_async_batch=do_async_batch
        )
        results.extend(result)

        ALL_TYPES = [ptype for types in GROUPED_TYPES.values() for ptype in types]
        paraphrase_type_tqdm = tqdm(ALL_TYPES)

        for paraphrase_type in paraphrase_type_tqdm:
            paraphrase_type_tqdm.set_description(
                f"Processing paraphrase type: {paraphrase_type}"
            )
            new_instruction = paraphrase_prompt(original_instruction, paraphrase_type)
            result = run_llm(
                llmchain=llmchain,
                instruction=new_instruction,
                main_dataset_inputs=main_dataset_inputs,
                main_dataset_outputs=main_dataset_outputs,
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                limit_examples=limit_examples,
                paraphrase_type=paraphrase_type,
                domains=domains,
                reasoning=reasoning,
                task_name=task_name,
                llmid=llmid,
                original_instruction=original_instruction,
                max_concurrency=max_concurrency,
                do_async_batch=do_async_batch
            )
            results.extend(result)

        # Store the results as csv for each task
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)


def run_tasks_for_llms(tasks, llms, limit_examples, max_concurrency, stop=None, do_async_batch=False):
    # Prompt
    prompt = PromptTemplate(
        input_variables=[
            "instruction",
            "positive_examples",
            "negative_examples",
            "example",
        ],
        template="Instruction: {instruction}\n\nPositive examples.\n{positive_examples}\n\nNegative examples.\n{negative_examples}\n\nNow, predict the output for the following example.\nInput: {example}\nOutput:"
    )

    for llmid, llm in llms.items():
        llmchain = LLMChain(llm=llm, prompt=prompt)
        run_tasks(
            tasks, llmchain, prompt, limit_examples=limit_examples, llmid=llmid, max_concurrency=max_concurrency, stop=stop, do_async_batch=do_async_batch
        )


if __name__ == "__main__":
    limit_examples = 200 # The maximum number of examples per task to be sampled, None for all
    limit_domains = None  # The hard max number of domains, None for all
    min_tasks_per_domain = 10  # This is the minimum number of tasks are needed for a domain to be considered
    num_sample_tasks_per_domain = 5 # This is how many tasks per domain are sampled
    max_concurrency = 200 # For R+ set to 200, for smaller models (<=70B) set to 500
    use_tgi = False # LLMs not trained for instructions
    use_tgi_chat = True # LLMs trained for instructions
    stop = None # special tokens to stop the generation for
    sampled_task_file = "data/sample_tasks.txt" # List of sampled tasks, None to sample new tasks
    do_async_batch = False
    tgi_server_url = os.environ.get("TGI_SERVER_URL", "http://127.0.0.1:8080")

    if sampled_task_file:
        with open(sampled_task_file, "r") as f:
            tasks = f.read().splitlines()
    else:
        # Load the tasks
        tasks = []
        tasks_per_domain = count_tasks_per_domain_and_language("English", write_csv=True)
        domains_with_n_tasks = get_domains_with_n_tasks(
            tasks_per_domain, min_tasks_per_domain
        )
        for domain in domains_with_n_tasks[:limit_domains]:
            tasks.extend(sample_tasks_from_domain(domain, num_sample_tasks_per_domain))
    logging.warning(f"Running {len(tasks)} tasks.")
    print("All tasks:\n" + "\n".join(tasks))

    # All LLMs to run
    llms = {}

    # HF TGI (no Chat)
    if use_tgi:
        tgi_info = requests.get(f"{tgi_server_url}/info")
        if tgi_info.status_code != 200:
            logging.error("HuggingFace TGI Server not reachable.")
            raise Exception("HuggingFace TGI Server not reachable.")
        tgi_info = tgi_info.json()
        tgi_model_name = tgi_info["model_id"]
        hf_api_token = os.environ.get("HF_API_TOKEN")
        hf_llm = HuggingFaceEndpoint(
            endpoint_url=tgi_server_url,
            huggingfacehub_api_token=hf_api_token,
            timeout=240,
        )  # type: ignore
        llms[tgi_model_name] = hf_llm
        do_async_batch = True

    # HF TGI Chat
    if use_tgi_chat:
        tgi_info = requests.get(f"{tgi_server_url}/info")
        if tgi_info.status_code != 200:
            logging.error("HuggingFace TGI Server not reachable.")
            raise Exception("HuggingFace TGI Server not reachable.")
        tgi_info = tgi_info.json()
        tgi_model_name = tgi_info["model_id"]
        client = OpenAI(
            base_url=f"{tgi_server_url}/v1",
            api_key="-"
        )
        stop = None
        if "llama-3" in tgi_model_name.lower():
            stop = [
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
                "<|reserved_special_token|>"
            ]
        llm = HFTGIChat(client=client, stop=stop, model_name=tgi_model_name)
        llms[tgi_model_name] = llm
        do_async_batch = True

    # Run tasks
    model_names = " and ".join(llms.keys())
    logging.warning(f"Running tasks using the models: {model_names}")
    run_tasks_for_llms(tasks, llms, limit_examples, max_concurrency, stop=stop, do_async_batch=do_async_batch)
