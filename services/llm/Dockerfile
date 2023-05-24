# syntax=docker/dockerfile:1.2
FROM python:3.11-slim-bullseye

WORKDIR /workspace

# Allows python to stream logs rather than buffer them for output.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt install -y libpq-dev gcc

# Install pip packages
COPY services/llm/requirements.txt  .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy base files
COPY . .