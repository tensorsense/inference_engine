from pathlib import Path
import pandas as pd
import json
from typing import List

from core.common.config import EngineConfig
from core.common.types import Sample, Batch, LLMOutput


def load_qa(config: EngineConfig) -> pd.DataFrame:
    # read annotations from disk

    with config.qa_path.open("r") as f:
        qa = json.load(f)

    samples = []
    for entry in qa:
        for i, conversation in enumerate(entry["conversations"]):
            assert conversation[0]["from"] == "human"
            assert conversation[1]["from"] == "gpt"
            d = {
                "video": entry["video"],
                "messages": conversation,
                "question_id": f"{entry['id']}_{i}",
                "llm_prediction": entry.get('llm_predictions')[i], # None if not present, must be same length as conversations
                "ntokens": entry.get('ntokens'), # None if not present
                "proctime": entry.get('proctime'), # None if not present
            }
            samples.append(d)

    df = pd.DataFrame(samples)

    for _, row in df[df['llm_prediction'].notnull()].iterrows():
        llm_output = LLMOutput.from_sample(
            Sample(
                video_path=Path(config.video_path)
                .joinpath(row.video)
                .resolve()
                .as_posix(),
                text_prompt=row.messages[0]["value"],
                target_answer=row.messages[1]["value"],
                question_id=row.question_id,
            ),
            llm_prediction=row["llm_prediction"],
            ntokens=row["ntokens"],
            proctime=row["proctime"],
        )
        with config.llm_output_path.joinpath(f"{llm_output.question_id}.json").open(
            "w"
        ) as f:
            f.write(llm_output.model_dump_json())

    existing_outputs = [path.stem for path in config.llm_output_path.glob("*.json")]
    filtered_df = df[~df["question_id"].isin(existing_outputs)]
    return filtered_df


# prepare batches
def prepare_batches(config: EngineConfig, df: pd.DataFrame) -> List[Batch]:
    batches = []
    for start in range(0, len(df), config.batch_size):
        end = start + config.batch_size
        chunk = df.iloc[start:end]

        batch = Batch(
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
        )

        for _, row in chunk.iterrows():
            batch.inputs.append(
                Sample(
                    video_path=Path(config.video_path)
                    .joinpath(row.video)
                    .resolve()
                    .as_posix(),
                    text_prompt=row.messages[0]["value"],
                    target_answer=row.messages[1]["value"],
                    question_id=row.question_id,
                )
            )
        batches.append(batch)
    return batches
