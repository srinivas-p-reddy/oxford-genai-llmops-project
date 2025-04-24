# Makefile
# This makefile should be used to develop the app locally.
# Current suggested run order is:
#	1. install
# 	1. build-db (outside devcontainer)
# 	2. run-ingestion (inside devcontainer)
# 	3. run-app (inside devcontainer)
#   4. run-client (inside devcontainer)
# Define directories
SERVER_DIR=./server
# Generator model
GENERATOR_MODEL=tinyllama#llama3.1:8b

# Install dependencies
install:
	poetry install

# Run the FastAPI app
run-app:
	@echo "Starting RAG app..."
	@bash -c 'set -a && source .env && set +a && export PYTHONPATH=./server/src && poetry run uvicorn server.src.main:app --reload'



run-client:
	poetry run streamlit run client/streamlit_app.py

# Build the postgres db - don't run inside the devctontainer ... TODO: build logic to check if I'm in devcontainer.
build-db:
	 docker compose --env-file .env -f deploy/docker/postgres/docker-compose.yaml up --build

# Bring down the postgres db and then remove the config data.
remove-db:
	docker compose --env-file .env -f deploy/docker/postgres/docker-compose.yaml down
	rm -rf deploy/docker/postgres/data

# Download the paper metadata from arxiv using the public API.
download-data:
	poetry run python $(SERVER_DIR)/src/ingestion/arxiv_client.py

# Note that the below is ingestion of the downloaded jsons, not downloading of arxiv data.
# Note that this will trigger a download of hugging face embedding models if not already cached. TODO: replace with Bedrock.
run-ingestion:
	poetry run python $(SERVER_DIR)/src/ingestion/pipeline.py

# Lint the project (using Ruff)
lint:
	poetry run ruff $(SERVER_DIR)

# Format the code (using black)
format:
	poetry run black $(SERVER_DIR)

# Run tests
test:
	poetry run pytest

# Clean up build artifacts (if any)
clean:
	find . -type f -name '*.pyc' -delete


# After a test this doesn't work outside the container, you need to build postgres first and then
# run the ingestion seperately just now.
# TODO: allow for build and ingestion outside the container in one command.
#run-ingestion-and-build-db: build-db
#	poetry run python $(SERVER_DIR)/src/ingestion/pipeline.py


# Note: as things stand I need to run build-ollama and build-db outside the dev container.
# I then need to run other commands inside the dev container (such as run-ingestion).
# TODO: Fix this.
# Current run order is:
# 1. build-ollama (outside devcontainer)
# 2. build-db (outside devcontainer)
# 3. run-ingestion (inside devcontainer)
# 4. run-app (inside devcontainer)

# Build Ollama server - refactor on 13/2/2025, not needed.
# build-ollama:
# 	docker run -d -v ollama:/root/.ollama -p 11434:11434 --memory="5g" --name ollama ollama/ollama
# 	docker exec -it ollama ollama run $(GENERATOR_MODEL)

# # Remove Ollama server
# remove-ollama:
# 	docker stop /ollama
# 	docker rm /ollama



# TODO: Fix the fact I need to build-db outside the dev container and then run other commands inside it.
# deploy-local: donwload-data run-ingestion run-app
# 	echo "Local deployment complete. Run 'make remove-db' to clean up the database."
