# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lemonade SDK is a Python-based framework for serving local LLMs with GPU and NPU acceleration. It provides OpenAI-compatible APIs and supports both GGUF and ONNX models.

## Development Setup

### Installation
```bash
# Install for development with all dependencies
pip install -e .[dev,oga-cpu]

# Install specific backends
pip install -e .[oga-ryzenai]  # For NPU support
pip install -e .[oga-cpu]      # For CPU-only ONNX
```

### Key Commands

#### Running Tests
```bash
# Run unit tests
python test/llm_api.py
python test/server_cli.py
python test/server_unit.py

# Run example scripts for integration testing
python examples/api_basic.py
python examples/api_streaming.py
```

#### Code Quality
```bash
# Format code with Black (required)
black src/

# Lint with PyLint
pylint src/lemonade --rcfile .pylintrc --disable E0401
pylint examples --rcfile .pylintrc --disable E0401,E0611,F0010 --jobs=1 -v
```

#### CLI Commands
```bash
# Main Lemonade CLI
lemonade -m -i <model> huggingface-load llm-prompt -p "text" --max-new-tokens 10

# Server CLI (from source)
lemonade-server-dev run <model-name>
lemonade-server-dev pull <model-name>
lemonade-server-dev list
lemonade-server-dev serve --model <model-name>

# Developer shorthand
lsdev run <model-name>
```

## Architecture

### Core Components

**lemonade/** - Main SDK package
- `api.py` - High-level OpenAI-compatible API interface
- `cli.py` - Main CLI entry point and tool orchestration
- `sequence.py` - Tool chaining and workflow management
- `state.py` - Global state management for tools

**lemonade/tools/** - Inference engine implementations
- `huggingface/` - HuggingFace transformers integration
- `oga/` - ONNX Runtime GenAI integration
- `llamacpp/` - LlamaCpp integration for GGUF models
- `server/` - OpenAI-compatible server implementation
- `accuracy.py`, `mmlu.py`, `humaneval.py` - Evaluation tools

**lemonade_server/** - Server-specific functionality
- `cli.py` - Server CLI implementation
- `model_manager.py` - Model download and management
- `server_models.json` - Pre-configured model registry
- `settings.py` - Server configuration management

**lemonade/common/** - Shared utilities
- `inference_engines.py` - Unified interface for different backends
- `system_info.py` - Hardware detection (GPU, NPU, CPU)
- `network.py` - Download and networking utilities

### Inference Backends

The system supports multiple inference backends selected based on model format and available hardware:

1. **ONNX Models** (`.onnx` extension)
   - OGA CPU: CPU-only inference
   - OGA RyzenAI: NPU acceleration on AMD Ryzen AI
   - OGA iGPU/Hybrid: GPU acceleration

2. **GGUF Models** (`.gguf` extension)
   - LlamaCpp with various backends (CPU, Vulkan, ROCm)

3. **HuggingFace Models**
   - Direct transformers library integration

### Server Architecture

The server provides an OpenAI-compatible API at `http://localhost:11434/v1/` with:
- `/v1/chat/completions` - Chat completion endpoint
- `/v1/models` - List available models
- Model management via CLI (pull, delete, list)
- Built-in web UI for testing

## Key Patterns

### Tool System
All functionality is implemented as "tools" that can be chained together:
- Tools inherit from `Tool` base class
- Tools communicate via shared `State` object
- Tools are composed into sequences via CLI or programmatically

### Model Loading
Models are loaded through a unified interface:
1. Check model format (ONNX vs GGUF)
2. Detect available hardware (NPU, GPU, CPU)
3. Select appropriate backend
4. Load with backend-specific optimizations

### Error Handling
- Custom exceptions in `common/exceptions.py`
- Server-specific errors in `lemonade_server/cli.py`
- Hardware compatibility checks before loading

## Testing Approach

1. **Unit Tests** (`test/` directory)
   - Test individual components and APIs
   - Mock external dependencies where needed

2. **Integration Tests** (`examples/` directory)
   - End-to-end workflow testing
   - API compatibility verification

3. **CI/CD** (`.github/workflows/`)
   - Automated testing on Ubuntu and Windows
   - Black formatting enforcement
   - PyLint code quality checks

## Important Notes

- Always use Black formatter before committing
- The server runs on port 11434 by default (Ollama-compatible)
- Models are cached in `~/.lemonade/` directory
- Windows installer available for end-users
- Python 3.10-3.13 supported