# syntax=docker/dockerfile:1
FROM python:3.9-slim

ARG BUILD_DATE

LABEL org.opencontainers.image.title="Load Tests for ML Models"
LABEL org.opencontainers.image.description="Load tests for machine learning models."
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/schmidtbri/load-tests-for-ml-models"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="python:3.9-slim"

WORKDIR /service

ARG USERNAME=service-user
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update

# create a user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install --assume-yes --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# installing git because we need to install the model package from it's own github repository
RUN apt-get install --assume-yes --no-install-recommends git

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# installing dependencies first in order to speed up build by using cached layers
COPY ./service_requirements.txt ./service_requirements.txt
RUN pip install -r service_requirements.txt

COPY ./configuration ./configuration
COPY ./LICENSE ./LICENSE

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

USER $USERNAME
