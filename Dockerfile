FROM python:3.6
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
# COPY nltk_download.py ./
RUN python nltk_download.py
CMD python3 api.py

