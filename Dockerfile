FROM mpioperator/tensorflow-benchmarks:latest

USER root

RUN echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    python3 -m pip install --upgrade pip setuptools && \
    pip3 install torch mpi4py

RUN mkdir -p /scripts


COPY mpi_torch.py /scripts/mpi_torch.py


# Set the working directory
WORKDIR /scripts


CMD ["/bin/bash"]
