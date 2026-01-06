# Docker image configuration
IMAGE_NAME=egowalk-extraction
IMAGE_TAG=latest
CONTAINER_NAME=egowalk-extraction

# Build Docker image with current user's UID and GID
build_docker:
	docker build --rm \
		--build-arg USER_UID=$(shell id -u) \
		--build-arg USER_GID=$(shell id -g) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) .


run_docker:
	docker run --rm -it \
		--gpus all \
		--ipc host \
		--privileged \
		--network host \
		-v $(shell readlink -f dir_links/raw):/home/captain/data/raw:ro \
		-v $(shell readlink -f dir_links/processed):/home/captain/data/processed \
		-v $(CURDIR):/home/captain/code \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(IMAGE_TAG) /bin/bash
