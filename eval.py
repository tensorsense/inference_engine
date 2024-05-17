import multiprocessing
import queue
import requests
import time
from tqdm import tqdm
import openai
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


ENDPOINTS = [
    "http://localhost:5000/predict",
]

BATCH_SIZE = 20
OPENAI_AZURE_DEPLOYMENT = "gpt-35-turbo-1106"
NUM_OPENAI_WORKERS = 5

VIDEO_PATH = Path("/data/data/MSRVTT_Zero_Shot_QA/videos/all")
GT_QUESTIONS_PATH = Path("/data/data/MSRVTT_Zero_Shot_QA/test_q1000.json")
GT_ANSWERS_PATH = Path("/data/data/MSRVTT_Zero_Shot_QA/test_a.json")


# Get the current date and time
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

OUTPUT_PATH = Path(f"eval_results_{timestamp}")

OUTPUT_PATH.mkdir(parents=True, exist_ok=False)
llm_output_path = OUTPUT_PATH.joinpath("llm_output.jsonl")
openai_output_path = OUTPUT_PATH.joinpath("openai_output.jsonl")
scores_path = OUTPUT_PATH.joinpath("scores.json")

# prepare batches
def prepare_batches(batch_size, temperature, max_new_tokens):
    # read annotations from disk
    q_df = pd.read_json(GT_QUESTIONS_PATH)
    a_df = pd.read_json(GT_ANSWERS_PATH)

    df = pd.merge(q_df, a_df, on="question_id")

    batches = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        chunk = df.iloc[start:end]

        batch = {"inputs": []}

        for _, sample in chunk.iterrows():
            batch["inputs"].append(
                {
                    "video_path": Path(VIDEO_PATH)
                    .joinpath(f"{sample.video_name}.mp4")
                    .resolve()
                    .as_posix(),
                    "text_prompt": sample.question,
                    "question_id": sample.question_id,
                    "target_answer": sample.answer,
                }
            )
        batch["temperature"] = temperature
        batch["max_new_tokens"] = max_new_tokens
        batches.append(batch)
    return batches


# Function for each worker to process batches
def local_worker(worker_id, endpoint, batch_queue, results_queue, progress_queue):
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

            results_queue.put(sample_set)

        # Indicate that the task is done
        batch_queue.task_done()

        # Update progress
        progress_queue.put(1)


# Function to process results and send them to the OpenAI API
def openai_worker(
    worker_id,
    results_queue,
    final_results_list,
    progress_queue,
):
    client = openai.AzureOpenAI()

    while True:
        try:
            # Get the next result from the queue
            result = results_queue.get(timeout=25)
        except queue.Empty:
            print(f"OpenAI Worker {worker_id}: No more results to process. Exiting.")
            break

        question = result["question"]
        answer = result["answer"]
        pred = result["model_prediction"]

        try:
            completion = client.chat.completions.create(
                model=OPENAI_AZURE_DEPLOYMENT,
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
            final_results_list.append(result | response_dict)
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


if __name__ == "__main__":
    # Shared batch list (queue)
    batch_queue = multiprocessing.JoinableQueue()

    # Example batches (replace with actual batches)
    batches = prepare_batches(
        batch_size=BATCH_SIZE, temperature=0.1, max_new_tokens=1024
    )

    # batches = batches[:5]

    # Populate the batch queue
    for batch in batches:
        batch_queue.put(batch)

    # Create a multiprocessing manager to handle shared data
    manager = multiprocessing.Manager()
    results_queue = multiprocessing.Queue()
    final_results_list = manager.list()

    # Progress queue for monitoring progress
    local_progress_queue = multiprocessing.Queue()
    openai_progress_queue = multiprocessing.Queue()

    # Create and start worker processes for the first stage
    local_workers = []
    for i, endpoint in enumerate(ENDPOINTS):
        local_worker_process = multiprocessing.Process(
            target=local_worker,
            args=(i, endpoint, batch_queue, results_queue, local_progress_queue),
        )
        local_workers.append(local_worker_process)
        local_worker_process.start()

    # Create and start progress monitor process for the combined progress
    local_progress_monitor_process = multiprocessing.Process(
        target=progress_monitor, args=(len(batches), 0, local_progress_queue)
    )
    local_progress_monitor_process.start()

    # Create and start worker processes for the second stage (OpenAI API processing)
    openai_workers = []
    for i in range(NUM_OPENAI_WORKERS):  # Adjust the number of OpenAI workers as needed
        openai_worker_process = multiprocessing.Process(
            target=openai_worker,
            args=(
                i,
                results_queue,
                final_results_list,
                openai_progress_queue,
            ),
        )
        openai_workers.append(openai_worker_process)
        openai_worker_process.start()

    # Create and start progress monitor process for the combined progress
    openai_progress_monitor_process = multiprocessing.Process(
        target=progress_monitor,
        args=(len(batches) * BATCH_SIZE, 1, openai_progress_queue),
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
    final_results = list(final_results_list)

    with openai_output_path.open("w") as f:
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
