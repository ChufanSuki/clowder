FROM chufansuki:clowder:latest
# First step: install SC2 
RUN wget -O ~/sc2.zip http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && \
    unzip -P iagreetotheeula ~/sc2.zip -d ~/ && \
    rm ~/sc2.zip

## hack: fix the error message
## ffmpeg: ~/StarCraftII/Libs/libstdc++.so.6: version `GLIBCXX_3.4.26'
## but make sure you have ffmpeg installed
RUN  mv ~/StarCraftII/Libs/libstdc++.so.6 ~/StarCraftII/Libs/libstdc++.so.6.temp

# Second step: install pysc2 
RUN git clone https://github.com/vwxyzjn/gym-pysc2
RUN cd gym-pysc2

# Optional: construct virtualenv via `pyenv`
RUN pyenv install -s $(sed "s/\/envs.*//" .python-version)
RUN pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
RUN pip install --upgrade pip wheel

# install dependencies for running the training script
RUN pip install -r requirements.txt