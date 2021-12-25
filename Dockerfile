FROM ubuntu:focal@sha256:7cc0576c7c0ec2384de5cbf245f41567e922aab1b075f3e8ad565f508032df17

RUN apt-get update && \
    apt-get install -y python3-dev python3-pip git build-essential

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

CMD [ "zsh" ]