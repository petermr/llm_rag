# LocalAIAgentWithRAG

⏳ Timestamps ⏳
00:00 | Video Overview
00:34 | Project Demo
02:02 | Python Setup/Installation
04:33 | Ollama Setup
07:14 | GitHub Copilot
08:22 | Local LLM Usage
14:52 | Vector Store Database Setup
23:24 | Connecting LLM & Vector Store


# 🧠 LLM RAG Setup Guide

This guide walks you through setting up the `llm_rag` repository with a Python environment and Docker-based Ollama.

---

## 🔧 Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/petermr/llm_rag.git
   cd llm_rag

2. Create and Activate a Virtual Environment
    python3 -m venv venv
    source venv/bin/activate
3. Install Python Requirements
    pip3 install -r requirements.txt
4. Pull the Ollama Docker Image
    docker pull ollama/ollama
5. Run Ollama in Docker
    docker run -d --name ollama \
    -p 11434:11434 \
    -v ollama:/root/.ollama \
    ollama/ollama
6. Pull Required Models Inside the Container
    docker exec -it ollama ollama pull mxbai-embed-large
    docker exec -it ollama ollama pull llama3.2
7. You can list available models with:
    docker exec -it ollama ollama list
8. python3 main.py
 

