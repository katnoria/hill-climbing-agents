FROM python:3.6.6-slim-stretch

ENV PIP_PACKAGES="\
        pyyaml \
        cffi \
        h5py \
        numpy \
        pandas \
        matplotlib \
        jupyter \
        notebook \
        ipython \
        gym" \        
    PYTHON_VERSION=3.6.6 \
    PATH=/usr/local/bin:$PATH \
    PYTHON_PIP_VERSION=9.0.1 \
    LANG=C.UTF-8

RUN set -ex; \    
    apt-get update -y; \
    apt-get upgrade -y; \
    # ln -s /usr/bin/python3 /usr/bin/python; \
    # ln -s /usr/bin/python3-config /usr/bin/python-config; \
    # ln -s /usr/bin/pip3 /usr/bin/pip; \
    pip install -U -v setuptools wheel; \
    pip install -U -v ${PIP_PACKAGES}; \    
    # apt-get remove --purge --auto-remove -y ${BUILD_PACKAGES}; \
    apt-get clean; \
    apt-get autoclean; \
    apt-get autoremove -y;

EXPOSE 8080

RUN mkdir -p /app

COPY . /app

WORKDIR /app

ADD start.sh /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]