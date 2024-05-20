from pathlib import Path
import pandas as pd
import json
from typing import List, Dict

from core.common.config import EngineConfig


def load_qa(config: EngineConfig) -> pd.DataFrame:
    # read annotations from disk

    with config.qa_path.open("r") as f:
        qa = json.load(f)

    samples = []
    for entry in qa:
        for i, conversation in enumerate(entry["conversations"]):
            assert conversation[0]["from"] == "human"
            assert conversation[1]["from"] == "gpt"

            samples.append(
                {
                    "video": entry["video"],
                    "messages": conversation,
                    "question_id": f"{entry['video'][:-4]}_{i}",
                }
            )

    df = pd.DataFrame(samples)

    existing_outputs = [path.stem for path in config.llm_output_path.glob("*.json")]
    filtered_df = df[~df["question_id"].isin(existing_outputs)]
    return filtered_df


# prepare batches
def prepare_batches(config: EngineConfig, df: pd.DataFrame) -> List[Dict]:
    batches = []
    for start in range(0, len(df), config.batch_size):
        end = start + config.batch_size
        chunk = df.iloc[start:end]

        batch = {"inputs": []}

        for _, sample in chunk.iterrows():
            batch["inputs"].append(
                {
                    "video_path": Path(config.video_path)
                    .joinpath(sample.video)
                    .resolve()
                    .as_posix(),
                    "text_prompt": sample.messages[0]["value"],
                    "question_id": sample.question_id,
                    "target_answer": sample.messages[1]["value"],
                }
            )
        batch["temperature"] = config.temperature
        batch["max_new_tokens"] = config.max_new_tokens
        batches.append(batch)
    return batches
