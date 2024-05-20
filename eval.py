import multiprocessing
import queue
import requests
from tqdm import tqdm
import openai
from pathlib import Path
import pandas as pd
import json
import yaml
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


# Get the current date and time
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_PATH = Path(f"eval_results_{timestamp}")


def set_up_output_paths(output_path: Path):

    llm_output_path = output_path.joinpath("llm_output")
    openai_output_path = output_path.joinpath("openai_output")
    final_result_path = output_path.joinpath("final_result.jsonl")
    scores_path = output_path.joinpath("scores.json")

    llm_output_path.mkdir(exist_ok=True, parents=True)
    openai_output_path.mkdir(exist_ok=True, parents=True)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=False)

    return {
        "llm_output_path": llm_output_path,
        "openai_output_path": openai_output_path,
        "final_result_path": final_result_path,
        "scores_path": scores_path,
    }


# prepare batches
def prepare_batches(
    batch_size,
    temperature,
    max_new_tokens,
    qa_path,
    video_path,
    llm_output_path,
):
    # read annotations from disk

    with qa_path.open("r") as f:
        qa = json.load(f)
    
    samples = []
    for entry in qa:
        for i, conversation in enumerate(entry["conversations"]):
            assert conversation[0]["from"] == "human"
            assert conversation[1]["from"] == "gpt"

            samples.append({
                "video": entry["video"],
                "messages": conversation,
                "question_id": f"{entry['video'][:-4]}_{i}"
            })

    df = pd.DataFrame(samples)

    existing_outputs = [path.stem for path in llm_output_path.glob("*.json")]
    filtered_df = df[~df["question_id"].isin(existing_outputs)] 

    batches = []
    for start in range(0, len(filtered_df), batch_size):
        end = start + batch_size
        chunk = filtered_df.iloc[start:end]

        batch = {"inputs": []}

        for _, sample in chunk.iterrows():
            batch["inputs"].append(
                {
                    "video_path": Path(video_path)
                    .joinpath(sample.video)
                    .resolve()
                    .as_posix(),
                    "text_prompt": sample.messages[0]["value"],
                    "question_id": sample.question_id,
                    "target_answer": sample.messages[1]["value"],
                }
            )
        batch["temperature"] = temperature
        batch["max_new_tokens"] = max_new_tokens
        batches.append(batch)
    return batches, len(filtered_df)


# Function for each worker to process batches
def local_worker(
    worker_id, endpoint, batch_queue, results_queue, progress_queue, llm_output_path
):
    while True:
        try:
            # Get the next batch from the queue
            batch = batch_queue.get(timeout=10)  # Timeout to avoid infinite blocking
        except queue.Empty:
            print(f"Worker {worker_id}: No more batches to process. Exiting.")
            break
        # Process the batch (simulate by sending to an LLM endpoint)
        response = requests.post(endpoint, json=batch)

        if response.status_code == 200:
            result = response.json()
            # results_list.append(result)
            # print(f"Worker {worker_id}: Successfully processed batch: {batch}")
        else:
            print(f"Worker {worker_id}: Failed to process batch: {batch}")
            continue

        # Postprocess the outputs
        for sample, response in zip(batch["inputs"], result["predicted_texts"]):

            sample_set = {
                "id": sample["question_id"],
                "question": sample["text_prompt"],
                "answer": sample["target_answer"],
                "model_prediction": response,
                "ntokens": 0,
                "proctime": 0.0,
            }

            with llm_output_path.joinpath(f"{sample_set['id']}.json").open("w") as f:
                json.dump(sample_set, f)

            results_queue.put(sample_set)

        # Indicate that the task is done
        batch_queue.task_done()

        # Update progress
        progress_queue.put(1)


# Function to process results and send them to the OpenAI API
def openai_worker(
    worker_id,
    results_queue,
    final_results_dict,
    progress_queue,
    openai_azure_deployment,
    openai_output_path,
):
    client = openai.AzureOpenAI()

    while True:
        try:
            # Get the next result from the queue
            result = results_queue.get(timeout=25)
        except queue.Empty:
            print(f"OpenAI Worker {worker_id}: No more results to process. Exiting.")
            break

        if result["id"] in final_results_dict:
            progress_queue.put(1)
            continue

        question = result["question"]
        answer = result["answer"]
        pred = result["model_prediction"]

        try:
            completion = client.chat.completions.create(
                model=openai_azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer.",
                    },
                    {
                        "role": "user",
                        "content": "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a JSON string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide a JSON string. "
                        'For example, your response should look like this: {"pred": "yes", "score": 4.8}.',
                    },
                ],
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content
            response_dict = json.loads(response_message)
            final_result = result | response_dict

            with openai_output_path.joinpath(f"{final_result['id']}.json").open(
                "w"
            ) as f:
                json.dump(final_result, f)

            final_results_dict[final_result["id"]] = final_result
        except Exception as e:
            print(f"Failed to process: {e}")

        # Update progress
        progress_queue.put(1)


# Function to update progress bar
def progress_monitor(total_batches, position, progress_queue):
    with tqdm(total=total_batches, position=position, leave=True) as pbar:
        for _ in range(total_batches):
            progress_queue.get()
            pbar.update(1)


def load_results_from_disk(path: Path):
    """Loads results from disk into a list."""
    results = []
    for file_path in path.glob("*.json"):
        with file_path.open("r") as f:
            results.append(json.load(f))
    return results


if __name__ == "__main__":
    # Load the YAML configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    output_path = (
        Path(config["override_output_path"])
        if "override_output_path" in config
        else DEFAULT_OUTPUT_PATH
    )

    llm_output_path, openai_output_path, final_result_path, scores_path = list(
        set_up_output_paths(output_path).values()
    )

    endpoints = config["endpoints"]
    batch_size = config["batch_size"]
    openai_azure_deployment = config["openai_azure_deployment"]
    num_openai_workers = config["num_openai_workers"]

    video_path = Path(config["video_path"])
    qa_path = Path(config["qa_path"])

    temperature = config["temperature"]
    max_new_tokens = config["max_new_tokens"]

    # Shared batch list (queue)
    batch_queue = multiprocessing.JoinableQueue()

    # Example batches (replace with actual batches)
    batches, num_samples = prepare_batches(
        batch_size=batch_size,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        qa_path=qa_path,
        video_path=video_path,
        llm_output_path=llm_output_path,
    )

    batches = batches[:10]
    num_samples = len(batches) * batch_size

    num_batches = len(batches)

    # Populate the batch queue
    for batch in batches:
        batch_queue.put(batch)

    # Create a multiprocessing manager to handle shared data
    manager = multiprocessing.Manager()

    results_queue = multiprocessing.Queue()
    # Load existing tasks into the task queue
    existing_results = load_results_from_disk(llm_output_path)
    for result in existing_results:
        results_queue.put(result)

    final_results_dict = manager.dict()
    # Load existing results into the final results dictionary
    existing_final_results = load_results_from_disk(openai_output_path)
    for final_result in existing_final_results:
        final_results_dict[final_result["id"]] = final_result

    # Progress queue for monitoring progress
    local_progress_queue = multiprocessing.Queue()
    openai_progress_queue = multiprocessing.Queue()

    # Create and start worker processes for the first stage
    local_workers = []
    for i, endpoint in enumerate(endpoints):
        local_worker_process = multiprocessing.Process(
            target=local_worker,
            args=(
                i,
                endpoint,
                batch_queue,
                results_queue,
                local_progress_queue,
                llm_output_path,
            ),
        )
        local_workers.append(local_worker_process)
        local_worker_process.start()

    # Create and start progress monitor process for the combined progress
    local_progress_monitor_process = multiprocessing.Process(
        target=progress_monitor, args=(num_batches, 0, local_progress_queue)
    )
    local_progress_monitor_process.start()

    # Create and start worker processes for the second stage (OpenAI API processing)
    openai_workers = []
    for i in range(num_openai_workers):  # Adjust the number of OpenAI workers as needed
        openai_worker_process = multiprocessing.Process(
            target=openai_worker,
            args=(
                i,
                results_queue,
                final_results_dict,
                openai_progress_queue,
                openai_azure_deployment,
                openai_output_path,
            ),
        )
        openai_workers.append(openai_worker_process)
        openai_worker_process.start()

    # Create and start progress monitor process for the combined progress
    openai_progress_monitor_process = multiprocessing.Process(
        target=progress_monitor,
        args=(num_samples, 1, openai_progress_queue),
    )
    openai_progress_monitor_process.start()

    # Wait for all batches to be processed in the first stage
    batch_queue.join()

    # llm_results = list(results_queue)

    # with llm_output_path.open("w") as f:
    #     json.dump(llm_results, f)

    # Wait for all results to be processed by the OpenAI workers
    for worker_process in openai_workers:
        worker_process.join()

    # Ensure the progress monitor process finishes
    local_progress_monitor_process.join()
    openai_progress_monitor_process.join()

    # Terminate all worker processes for the first stage
    for worker_process in local_workers:
        worker_process.terminate()

    # Gather final results
    final_results = list(final_results_dict.values())

    with final_result_path.open("w") as f:
        json.dump(final_results, f)

    print("All batches processed.")
    # print("Final Results:", final_results)

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for result in tqdm(final_results):
        try:
            # Computing score
            count += 1
            score_match = result["score"]
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result["pred"]
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)

    with scores_path.open("w") as f:
        output_score = {
            "yes_count": yes_count,
            "no_count": no_count,
            "accuracy": accuracy,
            "average_score": average_score,
        }
        json.dump(output_score, f)
