FROM nvcr.io/nvidia/tensorflow:18.01-py3


# RUN apt-get update && apt-get install -y --no-install-recommends \
#   python-opencv \
#   python-matplotlib \
#   python-pillow 

RUN apt-get update
RUN apt-get install -y \
	python-skimage

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall

ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64

RUN ldconfig /usr/local/cuda/lib64

ADD src /src

WORKDIR /src

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py"]

