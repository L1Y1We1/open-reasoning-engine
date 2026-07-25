.PHONY: setup up down logs test lint ingest ask

setup:
	cp .env.example .env

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

test:
	pytest

lint:
	ruff check .

ingest:
	reasoning-engine ingest ./data

ask:
	reasoning-engine ask "$(Q)"

