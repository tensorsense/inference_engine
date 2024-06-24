from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional


class Sample(BaseModel):
    video_path: Path
    text_prompt: str
    target_answer: str
    question_id: str


class Batch(BaseModel):
    inputs: List[Sample] = Field(default_factory=list)
    temperature: float = Field(default=0.1)
    max_new_tokens: int = Field(default=64)


class BatchLLMPrediction(BaseModel):
    predicted_texts: List[str]
    ntokens: int
    proctime: float


class LLMOutput(Sample):
    llm_prediction: str
    ntokens: Optional[int]
    proctime: Optional[float]

    @classmethod
    def from_sample(
        cls, sample: Sample, llm_prediction: str, ntokens: Optional[int], proctime: Optional[float]
    ) -> "LLMOutput":
        return cls(
            video_path=sample.video_path,
            text_prompt=sample.text_prompt,
            target_answer=sample.target_answer,
            question_id=sample.question_id,
            llm_prediction=llm_prediction,
            ntokens=ntokens,
            proctime=proctime,
        )


class EvaluatorPrediction(BaseModel):
    pred: str
    score: int


class LLMOutputWithScore(LLMOutput):
    score: int
    pred: str

    @classmethod
    def from_llm_output(
        cls, llm_output: LLMOutput, score: int, pred: str
    ) -> "LLMOutputWithScore":
        return cls(
            video_path=llm_output.video_path,
            text_prompt=llm_output.text_prompt,
            target_answer=llm_output.target_answer,
            question_id=llm_output.question_id,
            llm_prediction=llm_output.llm_prediction,
            ntokens=llm_output.ntokens,
            proctime=llm_output.proctime,
            score=score,
            pred=pred,
        )


class Scores(BaseModel):
    yes_count: int
    no_count: int
    accuracy: float
    average_score: float
