# Deployment Guide

The service runs anywhere Python 3.12 runs. It starts in **offline mode** by
default and upgrades each component to its cloud backend when the corresponding
environment variables are present (see [`.env.example`](../.env.example)).

## 1. Docker (single container)

```bash
# workdir: project root
docker build -t legal-intelligence:latest -f deployment/Dockerfile .
docker run -p 8000:8000 legal-intelligence:latest
# open http://localhost:8000/docs
```

## 2. Docker Compose (API + Neo4j + Redis + Celery worker)

```bash
docker compose -f deployment/docker-compose.yml up --build
# API:   http://localhost:8000/docs
# Neo4j: http://localhost:7474  (neo4j / testpassword)
```

To run the API **without** the production backends, comment out the
`environment:` block in `docker-compose.yml` (or remove the `neo4j`/`redis`
services) and the platform falls back to the in-memory graph and synchronous
processing.

## 3. Render.com (free tier)

[`render.yaml`](./render.yaml) is a Render Blueprint. Push the repo to GitHub,
create a new Blueprint service in Render, and the API deploys automatically.
Add cloud credentials (`NEO4J_URI`, `GROQ_API_KEY`, `AWS_BUCKET`, …) as
environment variables in the Render dashboard to enable production backends.

## 4. AWS connectivity

When `AWS_BUCKET` is set and `boto3` is installed
(`pip install -r requirements-cloud.txt`), uploaded contract PDFs and JSON
analysis reports are persisted to S3 instead of the local `storage/` directory.
Provide credentials via standard AWS mechanisms (`AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY`, an instance role, or `~/.aws/credentials`).
