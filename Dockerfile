FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get -y install xvfb ffmpeg git build-essential wget checkinstall python3-pip && \
    apt --fix-broken install 
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN pip install PyOpenGL PyOpenGL_accelerate
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
COPY clowder/ /clowder/