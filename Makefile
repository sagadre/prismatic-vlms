.PHONY: help clean check autoformat
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, ruff) and update in place - committing with pre-commit also does this."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix --show-fixes .


# [TRI Internal] Sagemaker Docker Build + Push to ECR
SAGEMAKER_PROFILE ?= default
SAGEMAKER_REGION ?= us-east-1

# Prismatic VLM Sagemaker Build
SAGEMAKER_VLM_NAME ?= prismatic-vlms
sagemaker-prismatic:
	@echo "[*] Building Prismatic VLMs Sagemaker Container =>> Pushing to AWS ECR"; \
      echo "[*] Verifying AWS ECR Credentials"; \
	  account=$$(aws sts get-caller-identity --query Account --output text --profile ${SAGEMAKER_PROFILE}); \
	  echo "    => Found AWS Account ID = $${account}"; \
	  fullname=$${account}.dkr.ecr.${SAGEMAKER_REGION}.amazonaws.com/${SAGEMAKER_VLM_NAME}:latest; \
  	  echo "    => Setting ECR Registry Path = $${fullname}"; \
  	  echo ""; \
  	  echo "[*] Rebuilding ${SAGEMAKER_VLM_NAME} Docker Image"; \
  	  echo "    => Retrieving Official AWS Sagemaker Base Image"; \
	  aws ecr get-login-password --region ${SAGEMAKER_REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com; \
	  echo "    => Building Image from Dockerfile"; \
  	  docker build -f scripts/sagemaker/vlm-training.Dockerfile -t ${SAGEMAKER_VLM_NAME} .; \
  	  docker tag ${SAGEMAKER_VLM_NAME} $${fullname}; \
  	  echo ""; \
  	  echo "[*] Pushing Image to ECR Path = $${fullname}"; \
  	  aws ecr get-login-password --region ${SAGEMAKER_REGION} | docker login --username AWS --password-stdin $${fullname}; \
  	  docker push $${fullname};