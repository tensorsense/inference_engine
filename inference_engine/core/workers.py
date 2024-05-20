import queue
import requests
from tqdm import tqdm
import openai
from pathlib import Path
import json

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
