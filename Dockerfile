FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get remove python3 && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get -y install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev \
    xvfb ffmpeg git build-essential python-opengl wget checkinstall && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tgz && \
    tar xzf Python-3.10.4.tgz && \
    cd Python-3.10.4 && \
    ./configure --enable-optimizations --prefix=/usr && \
    make install
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN pip install poetry --upgrade
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install
RUN poetry install --with atari
RUN poetry install --with pybullet

# install mujoco_py
RUN apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
RUN poetry install --with mujoco_py
RUN poetry run python -c "import mujoco_py"

COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# copy local files
COPY ./clowder /clowder