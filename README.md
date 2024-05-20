# Inference Engine for LLM Video Benchmarks

This repository contains an inference engine designed to quickly and efficiently run video-based large language model (LLM) benchmarks. The engine leverages parallelism to maximize resource usage and minimize compute time.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Prepare Batches](#prepare-batches)
  - [Run Inference](#run-inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/tensorsense/inference_engine.git
cd inference_engine
pip3 install -r requirements.txt
```

## Configuration

The engine requires a configuration file (`config.yaml`) to specify various parameters.
An example `config.yaml` file is included in the repository. You can use it as a template and modify it according to your requirements.

### API Keys

You need to create a `.env` file in the root directory of the repository and add your API keys to it. The `.env` file should look like this:

```
OPENAI_API_VERSION="2023-07-01-preview"
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Prepare Batches

The first step is to prepare batches of video-question pairs for processing. The `prepare_batches` function reads the input data and creates batches based on the configuration.

### Run Inference

Run the main script to start the inference process:

```bash
python3 eval.py
```

This will:
1. Set up output paths.
2. Load the configuration.
3. Prepare batches.
4. Start local workers for LLM inference.
5. Start OpenAI workers for evaluation.
6. Monitor progress and save results.

The engine uses multiprocessing to parallelize the processing of batches, significantly reducing the overall compute time.

## Results

After the inference and evaluation processes are completed, results will be saved in the specified output directory. The final results include detailed information about each question-answer pair, the model's prediction, and evaluation scores.

The following metrics are computed and saved:
- Average Score
- Accuracy
- Yes/No counts

These metrics provide insights into the performance of the evaluated models.

## Contributing

We welcome contributions to improve the inference engine. Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
