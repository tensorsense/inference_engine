from pathlib import Path
import yaml
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List

# Get the current date and time
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_PATH = Path(f"eval_results_{timestamp}").resolve()


class EngineConfig(BaseModel):
    endpoints: List[str]
    batch_size: int
    num_openai_workers: int

    openai_azure_deployment: str

    temperature: float
    max_new_tokens: int

    video_path: Path
    qa_path: Path

    override_output_path: Optional[Path] = Field(default=None)

    root_output_path: Optional[Path] = Field(default=None)
    llm_output_path: Optional[Path] = Field(default=None)
    openai_output_path: Optional[Path] = Field(default=None)
    final_result_path: Optional[Path] = Field(default=None)
    scores_path: Optional[Path] = Field(default=None)

    def setup_output_paths(self) -> None:
        self.root_output_path = (
            Path(self.override_output_path)
            if self.override_output_path is not None
            else DEFAULT_OUTPUT_PATH
        )

        self.llm_output_path = self.root_output_path.joinpath("llm_output")
        self.openai_output_path = self.root_output_path.joinpath("openai_output")
        self.final_result_path = self.root_output_path.joinpath("final_result.jsonl")
        self.scores_path = self.root_output_path.joinpath("scores.json")

        self.llm_output_path.mkdir(exist_ok=True, parents=True)
        self.openai_output_path.mkdir(exist_ok=True, parents=True)

        if not self.root_output_path.exists():
            self.root_output_path.mkdir(parents=True, exist_ok=False)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "EngineConfig":
        with yaml_path.open("r") as f:
            raw_config = yaml.safe_load(f)

        config = cls(
            endpoints=raw_config["endpoints"],
            batch_size=raw_config["batch_size"],
            openai_azure_deployment=raw_config["openai_azure_deployment"],
            num_openai_workers=raw_config["num_openai_workers"],
            video_path=Path(raw_config["video_path"]),
            qa_path=Path(raw_config["qa_path"]),
            temperature=raw_config["temperature"],
            max_new_tokens=raw_config["max_new_tokens"],
            override_output_path=(
                raw_config["override_output_path"]
                if "override_output_path" in raw_config
                else None
            ),
        )

        config.setup_output_paths()
        return config
