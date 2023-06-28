FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=true
WORKDIR /tmp
RUN apt-get update && apt-get install build-essential -y

# Correction of the following errors
# https://github.com/oobabooga/text-generation-webui/issues/1534#issuecomment-1555024722
RUN apt-get install gcc-11 g++-11 -y

COPY requirements.txt /tmp/requirements.txt
RUN CXX=g++-11 CC=gcc-11 pip install -r requirements.txt

WORKDIR /app
COPY . /app
ENTRYPOINT ["./babyagi.py"]
EXPOSE 8080
