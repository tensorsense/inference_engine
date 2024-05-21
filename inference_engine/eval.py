import multiprocessing
from tqdm import tqdm
from pathlib import Path
import json
from dotenv import find_dotenv, load_dotenv
from typing import Dict

_ = load_dotenv(find_dotenv())

from core.common.config import EngineConfig
from core.common.types import LLMOutput, LLMOutputWithScore, Scores
from core.data import load_qa, prepare_batches
from core.workers import local_worker, openai_worker, progress_monitor


def load_from_disk(path: Path, cls):
    """Loads results from disk into a list."""
    results = []
    for file_path in path.glob("*.json"):
        with file_path.open("r") as f:
            results.append(cls.model_validate_json(json.loads(f)))
    return results


def calculate_scores(scored_outputs_dict: Dict[str, LLMOutputWithScore]) -> Scores:
    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for llm_output_with_score in tqdm(scored_outputs_dict.values()):
        # Computing score
        count += 1
        score_sum += llm_output_with_score.score

        # Computing accuracy
        pred = llm_output_with_score.pred
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)

    return Scores(
        yes_count=yes_count,
        no_count=no_count,
        accuracy=accuracy,
        average_score=average_score,
    )


def main():
    # Read YAML configuration and set up output directories
    config = EngineConfig.from_yaml(
        Path(__file__).joinpath("../../config.yaml").resolve()
    )

    # Shared batch list (queue)
    batch_queue = multiprocessing.JoinableQueue()

    qa_df = load_qa(config)
    batches = prepare_batches(config, qa_df)

    # Important to set correctly, otherwise progress monitors will get stuck
    num_batches = len(batches)
    num_samples = len(qa_df)

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
                "llm_outputs_queue": llm_outputs_queue,
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
                "llm_outputs_queue": llm_outputs_queue,
                "scored_outputs_dict": scored_outputs_dict,
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

    # Wait for all results to be processed by the OpenAI workers
    for worker_process in openai_workers:
        worker_process.join()

    print("All workers quit.")

    # Ensure the progress monitor process finishes
    local_progress_monitor_process.join()
    openai_progress_monitor_process.join()

    print("Progress monitors quit.")

    # Terminate all worker processes for the first stage
    for worker_process in local_workers:
        worker_process.terminate()

    print("All batches processed.")

    scores = calculate_scores(scored_outputs_dict)
    print(scores)

    with config.scores_path.open("w") as f:
        f.write(scores.model_dump_json())


if __name__ == "__main__":
    main()
