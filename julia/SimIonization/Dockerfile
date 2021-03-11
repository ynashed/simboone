# Use an official Python runtime as a parent image
FROM julia:latest

MAINTAINER Youssef Nashed "ynashed@slac.stanford.edu"

ARG SCRATCH_VOLUME=/scratch
ENV SCRATCH_VOLUME=/scratch
RUN echo creating ${SCRATCH_VOLUME} && mkdir -p ${SCRATCH_VOLUME}
VOLUME ${SCRATCH_VOLUME}

WORKDIR /work

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

ADD . /work
