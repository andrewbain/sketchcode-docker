FROM python:3.6.2

RUN mkdir -p /home/project/sketchcode
WORKDIR /home/project/sketchcode
COPY requirements.txt /home/project/sketchcode
RUN pip install --no-cache-dir -r requirements.txt

COPY . /home/project/sketchcode