# Public Admistration Services Chatbot (Vietnam)

A chatbot system for Vietnamese Public Service Administration workflows.
It currently runs on Telegram and is designed to answer procedural questions, guide users through required steps, and retrieve relevant information from curated public service documents.

Important: This project is recommended to run on Linux or macOS. Windows is currently not supported.

## What It Does

- Handles citizen questions for public registration procedures on Telegram.
- Uses an LLM + retrieval pipeline to answer from a managed knowledge base.
- Supports document upload, indexing, versioning, and re-training through API endpoints.
- Stores conversation context and metadata for session-aware responses.
- Supports image-based assistance with OCR and highlight output for document guidance.

## Current Scope

- Channel: Telegram (active production-facing channel in this project).
- Domain: Vietnamese Public Service registration procedures.
- Knowledge base: Local training data under app/api/data/training_data.

## Tech Stack

- Backend API: FastAPI + SQLModel
- Chat orchestration: Rasa (core + actions)
- LLM and retrieval: LangChain + OpenAI embeddings
- Vector search: PostgreSQL + pgvector
- OCR support: Tesseract + Pillow
- Infra: Docker Compose + Makefile automation
- Tunneling for webhook exposure: ngrok

## High-Level Architecture

1. User sends a message or image in Telegram.
2. Telegram webhook hits the FastAPI endpoint.
3. The API normalizes request metadata and calls the LLM/retrieval layer.
4. Retrieved context + model output are returned as a Telegram response.
5. Rasa fallback/actions are used for conversational flow and integration.

Core services in docker-compose.yml:

- api: FastAPI service for domain logic and LLM endpoints.
- rasa-core: Rasa server.
- rasa-actions: custom Rasa actions.
- rasa-credentials: helper service for dynamic webhook/credentials setup.
- db: PostgreSQL with pgvector extension.
- ngrok: tunnel service for Telegram webhook callbacks.

## Quick Start (Docker)

### 1. Prerequisites

- Operating system: Linux or macOS (recommended). Windows is not supported.

- Docker + Docker Compose
- Python 3.9+
- Make
- OpenAI API key
- Telegram bot token
- ngrok auth token

### 2. Configure environment

Create your environment file in the project root and set required values:

```bash
cp .env-example .env
```

Required values typically include:

- OPENAI_API_KEY
- TELEGRAM_ACCESS_TOKEN
- TELEGRAM_BOTNAME
- NGROK_AUTH_TOKEN
- database variables (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)

### 3. Install and run

```bash
make install
make run
```

Useful commands:

```bash
make stop
make restart
make logs
make seed
```

## Local API Development

If you only want to work on the API service:

```bash
cd app/api
make install
make run
```

This path uses local virtualenv + uvicorn workflow from app/api/Makefile.

## API and Operations

- API docs: http://localhost:8888/docs
- Health check: http://localhost:8888/health
- Telegram webhook route: /webhooks/telegram/webhook

## Public Service Data

Domain content is organized by procedure under:

- app/api/data/training_data/birth_registration
- app/api/data/training_data/death_registration
- app/api/data/training_data/marriage_registration
- app/api/data/training_data/permanent_residence_registration
- app/api/data/training_data/id_card_issuance
- app/api/data/training_data/id_card_re-issuance
- app/api/data/training_data/guardianship_registration
- app/api/data/training_data/domestic_registration

You can extend these folders with additional Vietnamese public service procedures as needed.
