<div align="center">

# A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces

<a href="https://arxiv.org/abs/2602.03442"><img src="https://img.shields.io/badge/arXiv-2602.03442-b31b1b.svg" alt="arXiv"></a>
<a href="https://agentresearchlab.org/agents/a-rag/index.html#home"><img src="https://img.shields.io/badge/Website-A--RAG-blue" alt="Website"></a>
<a href="https://huggingface.co/datasets/Ayanami0730/rag_test"><img src="https://img.shields.io/badge/🤗_Datasets-A--RAG-yellow" alt="HuggingFace"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>

**If you find our project helpful, please give us a star ⭐ on GitHub!**

</div>

---

## 🚀 Quick Start

```bash
# 1. Install
git clone https://github.com/Ayanami0730/arag.git && cd arag
uv sync --extra full                  # or: pip install -e ".[full]"

# 2. Download benchmark datasets from HuggingFace
git clone https://huggingface.co/datasets/Ayanami0730/rag_test data --depth 1
rm -rf data/.git data/README.md

# 3. Build embedding index
#    We use Qwen3-Embedding-0.6B in our paper (https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
#    You can also use a local path: --model /path/to/Qwen3-Embedding-0.6B
uv run python scripts/build_index.py \
    --chunks data/musique/chunks.json \
    --output data/musique/index \
    --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda:0

# 4. Set environment variables
export ARAG_API_KEY="your-api-key"
export ARAG_BASE_URL="https://api.openai.com/v1"
export ARAG_MODEL="gpt-5-mini"

# 5. Run A-RAG agent
uv run python scripts/batch_runner.py \
    --config configs/example.yaml \
    --questions data/musique/questions.json \
    --output results/musique \
    --limit 10 --workers 5

# 6. Evaluate results
uv run python scripts/eval.py \
    --predictions results/musique/predictions.jsonl \
    --workers 5
```

> **Note**: Datasets hosted on [HuggingFace 🤗](https://huggingface.co/datasets/Ayanami0730/rag_test), reformatted from [Zly0523/linear-rag](https://huggingface.co/datasets/Zly0523/linear-rag) and [GraphRAG-Bench](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench) into a unified format.
>
> Don't have `uv`? Install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## ✨ News

- **[Feb 2026]** 📄 Paper released on [arXiv](https://arxiv.org/abs/2602.03442)
- **[Feb 2026]** 🚀 Initial code and evaluation suite released

---

## 📖 Overview

Frontier language models have demonstrated strong reasoning and long-horizon tool-use capabilities. However, existing RAG systems fail to leverage these capabilities. They still rely on two paradigms:
1. **Graph RAG**: Designing an algorithm that retrieves passages in a single shot and concatenates them into the model's input
2. **Workflow RAG**: Predefining a workflow and prompting the model to execute it step-by-step

Neither paradigm allows the model to participate in retrieval decisions, preventing efficient scaling with model improvements.

### Three Principles of Agentic RAG

We identify three key principles that define true agentic autonomy:
- **Autonomous Strategy**: The agent dynamically chooses retrieval strategies based on task characteristics
- **Iterative Execution**: The agent supports multi-round execution, adapting based on intermediate results
- **Interleaved Tool Use**: The agent follows a ReAct-like action→observation→reasoning loop

<div align="center">
<img src="assets/three-paradigms.png" alt="Three Paradigms Comparison" width="800">

*Comparison of three RAG paradigms. Only A-RAG satisfies all three principles, making it a truly agentic framework.*
</div>

### Our Solution: A-RAG

**A-RAG** is an **A**gentic **RAG** framework that exposes **hierarchical retrieval interfaces** directly to the model. A-RAG provides three retrieval tools: `keyword_search`, `semantic_search`, and `chunk_read`, enabling the agent to adaptively search and retrieve information across multiple granularities.

<div align="center">
<img src="assets/Framework.png" alt="A-RAG Framework" width="800">

*Overview of A-RAG framework. The agent iteratively uses hierarchical retrieval tools to gather information from the corpus and autonomously decides when to provide the final answer.*
</div>

### Key Features

- 🔍 **Hierarchical Retrieval**: Keyword-level, sentence-level, and chunk-level information access
- 🤖 **True Agentic Autonomy**: Autonomous strategy, iterative execution, and interleaved tool use
- 📈 **Test-Time Scaling**: Performance improves with increased compute resources
- ⚡ **Context Efficient**: Achieves superior accuracy with comparable or fewer retrieved tokens

---

## 📊 Main Results

Results (%) of baselines and A-RAG on benchmark datasets in terms of **LLM-Evaluation Accuracy (LLM-Acc)** and **Contain-Match Accuracy (Cont-Acc)**. Best results are in **bold**, second best are <u>underlined</u>.

### GPT-4o-mini Backbone

| Method | MuSiQue |  | HotpotQA |  | 2Wiki |  | Med. | Novel |
|--------|:-------:|:----:|:--------:|:----:|:-----:|:----:|:----:|:-----:|
|        | LLM | Cont | LLM | Cont | LLM | Cont | LLM | LLM |
| **Vanilla Baselines** |||||||||
| Direct Answer | 18.3 | 13.9 | 45.4 | 40.7 | 30.3 | 49.7 | 68.6 | 45.3 |
| Naive RAG | 38.6 | 36.1 | 74.5 | <u>72.9</u> | 42.6 | 59.0 | 75.3 | 68.5 |
| **Graph-RAG & Workflow RAG** |||||||||
| GraphRAG | 26.4 | 20.8 | 33.2 | 33.3 | 18.4 | 47.2 | 51.3 | 28.8 |
| HippoRAG2 | 40.6 | 38.4 | **80.7** | 69.7 | **64.7** | **68.5** | 72.0 | <u>70.1</u> |
| LinearRAG | 34.8 | 26.3 | 72.0 | 60.5 | <u>62.9</u> | <u>62.3</u> | 53.1 | 45.4 |
| FaithfulRAG | 28.8 | 22.6 | 60.5 | 52.5 | 38.8 | 38.1 | 42.5 | 33.3 |
| MA-RAG | 34.1 | 27.4 | 60.6 | 54.4 | 51.0 | 53.4 | 62.3 | 44.5 |
| RAGentA | 32.2 | 29.9 | 63.0 | 62.4 | 27.7 | 50.3 | 67.7 | 61.3 |
| **A-RAG (Ours)** |||||||||
| A-RAG (Naive) | <u>43.8</u> | <u>38.5</u> | 76.6 | 70.7 | 52.3 | 62.4 | <u>79.0</u> | 70.0 |
| A-RAG (Full) | **46.1** | **39.6** | <u>77.1</u> | **74.0** | 60.2 | 63.7 | **79.4** | **72.7** |

### GPT-5-mini Backbone

| Method | MuSiQue |  | HotpotQA |  | 2Wiki |  | Med. | Novel |
|--------|:-------:|:----:|:--------:|:----:|:-----:|:----:|:----:|:-----:|
|        | LLM | Cont | LLM | Cont | LLM | Cont | LLM | LLM |
| **Vanilla Baselines** |||||||||
| Direct Answer | 35.8 | 26.5 | 63.6 | 53.5 | 51.3 | 54.0 | 90.5 | 45.1 |
| Naive RAG | 52.8 | 48.7 | 81.2 | 79.5 | 50.2 | 66.5 | 86.1 | 70.6 |
| **Graph-RAG & Workflow RAG** |||||||||
| GraphRAG | 48.3 | 39.1 | 82.5 | 74.9 | 66.5 | 70.7 | 87.3 | <u>77.1</u> |
| HippoRAG2 | 61.7 | 52.5 | 84.8 | 75.0 | 82.0 | 79.7 | 78.2 | 54.3 |
| LinearRAG | 62.4 | 51.8 | 86.2 | 77.6 | <u>87.2</u> | <u>84.8</u> | 79.2 | 54.7 |
| FaithfulRAG | 52.9 | 52.8 | 76.9 | 75.3 | 51.8 | 56.6 | 75.4 | 60.7 |
| MA-RAG | 40.0 | 31.6 | 67.1 | 57.9 | 54.7 | 54.3 | 68.3 | 45.1 |
| RAGentA | 38.3 | 37.4 | 61.2 | 65.0 | 24.0 | 53.5 | 73.7 | 60.2 |
| **A-RAG (Ours)** |||||||||
| A-RAG (Naive) | <u>66.2</u> | <u>59.7</u> | <u>90.8</u> | <u>85.3</u> | 70.6 | 76.9 | <u>92.7</u> | 80.4 |
| A-RAG (Full) | **74.1** | **65.3** | **94.5** | **88.0** | **89.7** | **88.9** | **93.1** | **85.3** |

---

## 📁 Project Structure

```
arag/
├── src/arag/              # Main package
│   ├── core/              # Core modules
│   │   ├── config.py      # Configuration management
│   │   ├── context.py     # Agent context & state tracking
│   │   └── llm.py         # LLM client with cost tracking
│   ├── agent/             # Agent implementations
│   │   ├── base.py        # BaseAgent with ReAct loop
│   │   └── prompts/       # System prompts
│   └── tools/             # Retrieval tools
│       ├── keyword_search.py
│       ├── semantic_search.py
│       └── read_chunk.py
├── scripts/               # CLI scripts
│   ├── build_index.py     # Build embedding index
│   ├── batch_runner.py    # Batch processing
│   └── eval.py            # Evaluation
├── configs/               # Configuration examples
├── tests/                 # Test suite (gitignored, add your own tests)
├── .github/               # Issue templates
└── CITATION.cff           # Citation metadata
```

---

## 🔧 Hierarchical Retrieval Tools

A-RAG provides three retrieval tools that operate at different granularities:

### Keyword Search

- **Method**: Exact lexical matching (case-insensitive)
- **Best for**: Known entities, names, technical terms
- **Score**: `Score(chunk, keywords) = Σ count(k, chunk) × |k|`
- **No pre-indexing required**

### Semantic Search

- **Method**: Dense retrieval using sentence-level embeddings
- **Best for**: Conceptual queries, when exact wording is unknown
- **Score**: Cosine similarity between query and sentence embeddings
- **Requires pre-built index**

### Chunk Read

- **Method**: Retrieve full content of specified chunks
- **Strategy**: Read promising chunks identified by search, read adjacent chunks (±1) for context
- **Context Tracker**: Prevents redundant reading of already-accessed chunks

---

## 📚 Benchmarks & Datasets

### Supported Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| MuSiQue | Multi-hop QA (2-4 hops) | [HuggingFace](https://huggingface.co/datasets/StonyBrookNLP/musique) |
| HotpotQA | Multi-hop QA | [HuggingFace](https://huggingface.co/datasets/hotpot_qa) |
| 2WikiMultiHopQA | Multi-hop QA | [GitHub](https://github.com/Alab-NII/2WikiMultiHopQA) |
| GraphRAG-Bench | Graph RAG evaluation | [HuggingFace](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench) |

### Custom Data Format

Prepare your own corpus as a JSON file:

```json
["0:Document chunk content here...", "1:Another chunk..."]
```

### Full Evaluation Example

<details>
<summary>Click to expand full evaluation instructions</summary>

#### 1. Build Index

```bash
# Using HuggingFace model (auto-download)
uv run python scripts/build_index.py \
    --chunks data/musique/chunks.json \
    --output data/musique/index \
    --model Qwen/Qwen3-Embedding-0.6B \
    --device cuda:0

# Or using a local model path
uv run python scripts/build_index.py \
    --chunks data/musique/chunks.json \
    --output data/musique/index \
    --model /path/to/Qwen3-Embedding-0.6B \
    --device cuda:0
```

#### 2. Create Config File

Create `configs/test_musique.yaml`:

```yaml
llm:
  temperature: 0.0
  max_tokens: 16384
  reasoning_effort: "medium"

embedding:
  model: "Qwen/Qwen3-Embedding-0.6B"  # or local path
  device: "cuda:0"
  batch_size: 16

agent:
  max_loops: 15
  max_token_budget: 128000
  verbose: false

data:
  chunks_file: "data/musique/chunks.json"
  index_dir: "data/musique/index"
```

#### 3. Run Full Benchmark

```bash
export ARAG_API_KEY="your-api-key"
export ARAG_BASE_URL="https://api.openai.com/v1"
export ARAG_MODEL="gpt-5-mini"
export CUDA_VISIBLE_DEVICES=0

# Run all questions
uv run python scripts/batch_runner.py \
    --config configs/test_musique.yaml \
    --questions data/musique/questions.json \
    --output results/musique \
    --workers 10

# Evaluate
uv run python scripts/eval.py \
    --predictions results/musique/predictions.jsonl \
    --workers 10
```

</details>

---

## 🐍 Python API

```python
from arag import LLMClient, BaseAgent, ToolRegistry
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool

# Initialize LLM client
client = LLMClient(
    model="gpt-5-mini",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# Setup tools
tools = ToolRegistry()
tools.register(KeywordSearchTool(chunks_file="data/chunks.json"))
tools.register(SemanticSearchTool(
    chunks_file="data/chunks.json",
    index_dir="data/index",
    embedding_model="Qwen/Qwen3-Embedding-0.6B"
))
tools.register(ReadChunkTool(chunks_file="data/chunks.json"))

# Create agent
agent = BaseAgent(
    llm_client=client,
    tools=tools,
    max_loops=15,
    max_token_budget=128000
)

# Run query
result = agent.run("What is the capital of France?")
print(f"Answer: {result['answer']}")
print(f"Cost: ${result['total_cost']:.6f}")
print(f"Loops: {result['loops']}")
```

---

## 🗺️ Roadmap

- [ ] **Baseline Scripts**: Compatible scripts for all baseline methods (GraphRAG, HippoRAG2, LinearRAG, etc.)
- [ ] **Ablation Interfaces**: Complete interfaces for ablation studies (w/o keyword search, w/o semantic search, w/o chunk read)
- [ ] **Multi-Provider Support**: Native API support for Anthropic Claude and Google Gemini (currently only OpenAI-compatible APIs)
- [ ] **Additional Benchmarks**: Scripts for HotpotQA, 2WikiMQA, and GraphRAG-Bench evaluation
- [ ] **Visualization Tools**: Trajectory visualization and analysis tools

Contributions and feedback are welcome!

---

## 📝 Citation

If you use A-RAG in your research, please cite our paper:

```bibtex
@misc{du2026aragscalingagenticretrievalaugmented,
      title={A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces}, 
      author={Mingxuan Du and Benfeng Xu and Chiwei Zhu and Shaohan Wang and Pengyu Wang and Xiaorui Wang and Zhendong Mao},
      year={2026},
      eprint={2602.03442},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.03442}, 
}
```

---

## 📄 License

MIT License

---

<div align="center">

**[Paper](https://arxiv.org/abs/2602.03442)** | **[Website](https://agentresearchlab.org/agents/a-rag/index.html#home)** | **[GitHub](https://github.com/Ayanami0730/arag)**

</div>
