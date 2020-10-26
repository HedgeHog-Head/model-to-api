FROM python:3.6

# first layers should be dependency install so changes in code won't cause the build to
# start from scratch.
COPY requirements.txt /opt/program/requirements.txt

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_PATH="/opt/ml/model"

# Set up the program in the image
COPY titanic_analysis /opt/program
WORKDIR /opt/program

