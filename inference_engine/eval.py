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
from pydantic import BaseModel
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

from core.common.config import EngineConfig
from core.common.types import LLMOutput, LLMOutputWithScore
from core.data import load_qa, prepare_batches
from core.workers import local_worker, openai_worker, progress_monitor


def load_from_disk(path: Path, cls):
    """Loads results from disk into a list."""
    results = []
    for file_path in path.glob("*.json"):
        with file_path.open("r") as f:
            results.append(cls.model_validate_json(json.loads(f)))
    return results


def calculate_scores(final_results):
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

    with config.scores_path.open("w") as f:
        output_score = {
            "yes_count": yes_count,
            "no_count": no_count,
            "accuracy": accuracy,
            "average_score": average_score,
        }
        json.dump(output_score, f)


def main():
    # Read YAML configuration and set up output directories
    config = EngineConfig.from_yaml(
        Path(__file__).joinpath("../../config.yaml").resolve()
    )

    # Shared batch list (queue)
    batch_queue = multiprocessing.JoinableQueue()

    qa_df = load_qa(config)
    batches = prepare_batches(config, qa_df)
    batches = batches[:1]

    num_samples = len(qa_df)
    num_batches = len(batches)

    # Populate the batch queue
    for batch in batches:
        batch_queue.put(batch)

    # Create a multiprocessing manager to handle shared data
    manager = multiprocessing.Manager()

    llm_outputs_queue = multiprocessing.Queue()
    # Load existing tasks into the task queue
    for file_path in config.llm_output_path.glob("*.json"):
        with file_path.open("r") as f:
            llm_outputs_queue.put(LLMOutput.model_validate_json(f.read()))

    scored_outputs_dict = manager.dict()
    # Load existing results into the final results dictionary
    for file_path in config.openai_output_path.glob("*.json"):
        with file_path.open("r") as f:
            scored_output = LLMOutputWithScore.model_validate_json(f.read())
            scored_outputs_dict[scored_output.question_id] = scored_output

    # Progress queue for monitoring progress
    local_progress_queue = multiprocessing.Queue()
    openai_progress_queue = multiprocessing.Queue()

    # Create and start worker processes for the first stage
    local_workers = []
    for i, endpoint in enumerate(config.endpoints):
        local_worker_process = multiprocessing.Process(
            target=local_worker,
            kwargs={
                "worker_id": i,
                "endpoint": endpoint,
                "batch_queue": batch_queue,
                "results_queue": llm_outputs_queue,
                "progress_queue": local_progress_queue,
                "llm_output_path": config.llm_output_path,
            },
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
    for i in range(config.num_openai_workers):
        openai_worker_process = multiprocessing.Process(
            target=openai_worker,
            kwargs={
                "worker_id": i,
                "results_queue": llm_outputs_queue,
                "final_results_dict": scored_outputs_dict,
                "progress_queue": openai_progress_queue,
                "openai_azure_deployment": config.openai_azure_deployment,
                "openai_output_path": config.openai_output_path,
            },
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

    # # Gather final results
    # final_results = list(scored_outputs_dict.values())

    # with config.final_result_path.open("w") as f:
    #     json.dump(final_results, f)

    # print("All batches processed.")
    # # print("Final Results:", final_results)

    # calculate_scores(final_results)


if __name__ == "__main__":
    main()
