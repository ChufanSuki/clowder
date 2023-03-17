FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update -y && \
    apt install -y software-properties-common && \
    apt-get update -y && \
    apt-get upgrade -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y &&  \
    apt-get -y install python3.9 xvfb ffmpeg git build-essential python-opengl wget checkinstall python3-pip
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN python --version

# install python dependencies
RUN pip install poetry --upgrade
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry env use 3.9
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