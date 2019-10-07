FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

# Config standard user
ARG USERNAME=pytorch

# Or your actual UID, GID on Linux if not the default 1000
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    #
    # [Optional] Add sudo support
    apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    rm -rf /var/lib/apt/lists/*

# Technically optional
ENV HOME /home/$USERNAME

# Config VS Code server directory
RUN mkdir "${HOME}/.vscode-server" && \
  chown -R ${USERNAME}:${USERNAME} "${HOME}/.vscode-server"

RUN conda install python==3.7
    
# Set the default user
USER $USERNAME

# Set the default shell to bash instead of sh
ENV SHELL /bin/bash

# Install python dependencies
ADD --chown=pytorch:pytorch requirements.txt .
RUN pip install --user -r requirements.txt && \
    rm requirements.txt

CMD ["/bin/bash"]