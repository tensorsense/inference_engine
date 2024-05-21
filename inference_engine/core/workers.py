import queue
import requests
from tqdm import tqdm
import openai
from pathlib import Path
import json
from multiprocessing import Queue

from pathlib import Path
from typing import Dict

from core.common.types import (
    Batch,
    BatchLLMPrediction,
    LLMOutput,
    EvaluatorPrediction,
    LLMOutputWithScore,
)


# Function for each worker to process batches
def local_worker(
    worker_id: int,
    endpoint: str,
    batch_queue: Queue,
    results_queue: Queue,
    progress_queue: Queue,
    llm_output_path: Path,
):
    while True:
        try:
            # Get the next batch from the queue
            batch: Batch = batch_queue.get(timeout=10)  # Timeout to avoid infinite blocking
        except queue.Empty:
            print(f"Worker {worker_id}: No more batches to process. Exiting.")
            break

        # Process the batch (simulate by sending to an LLM endpoint)
        response = requests.post(endpoint, data=batch.model_dump_json())

        if response.status_code == 200:
            batch_llm_prediction = BatchLLMPrediction.model_validate(response.json())
        else:
            print(f"Worker {worker_id}: Failed to process batch: {batch}")
            continue

        # Postprocess the outputs
        for sample, llm_prediction in zip(
            batch.inputs, batch_llm_prediction.predicted_texts
        ):
            llm_output = LLMOutput.from_sample(
                sample=sample,
                llm_prediction=llm_prediction,
                ntokens=batch_llm_prediction.ntokens,
                proctime=batch_llm_prediction.proctime,
            )

            with llm_output_path.joinpath(f"{llm_output.question_id}.json").open(
                "w"
            ) as f:
                f.write(llm_output.model_dump_json())

            results_queue.put(llm_output)

        # Indicate that the task is done
        batch_queue.task_done()

        # Update progress
        progress_queue.put(1)


# Function to process results and send them to the OpenAI API
def openai_worker(
    worker_id: int,
    results_queue: Queue,
    final_results_dict: Dict[str, LLMOutputWithScore],
    progress_queue: Queue,
    openai_azure_deployment: str,
    openai_output_path: Path,
):
    client = openai.AzureOpenAI()

    while True:
        try:
            # Get the next result from the queue
            llm_output = results_queue.get(timeout=25)
        except queue.Empty:
            print(f"OpenAI Worker {worker_id}: No more results to process. Exiting.")
            break

        if llm_output.question_id in final_results_dict:
            progress_queue.put(1)
            continue

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
                        f"Question: {llm_output.text_prompt}\n"
                        f"Correct Answer: {llm_output.target_answer}\n"
                        f"Predicted Answer: {llm_output.llm_prediction}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a JSON string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide a JSON string. "
                        'For example, your response should look like this: {"pred": "yes", "score": 4.8}.',
                    },
                ],
            )
            # Convert response to a Python dictionary.
            response_message = completion.choices[0].message.content

            evaluator_predition = EvaluatorPrediction.model_validate(
                json.loads(response_message)
            )
            # final_result = result | response_dict

            llm_output_with_score = LLMOutputWithScore.from_llm_output(
                llm_output=llm_output,
                score=evaluator_predition.score,
                pred=evaluator_predition.pred,
            )

            with openai_output_path.joinpath(
                f"{llm_output_with_score.question_id}.json"
            ).open("w") as f:
                f.write(llm_output_with_score.model_dump_json())

            final_results_dict[llm_output_with_score.question_id] = (
                llm_output_with_score
            )
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
