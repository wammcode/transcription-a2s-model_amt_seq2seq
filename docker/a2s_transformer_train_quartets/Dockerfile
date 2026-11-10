FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Source, corpus and outputs are mounted by p1.slurm.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/a2s-matplotlib \
    NUMBA_CACHE_DIR=/tmp/a2s-numba

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/a2s-requirements.txt
RUN python -m pip install --no-cache-dir \
        torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && \
    python -m pip install --no-cache-dir -r /tmp/a2s-requirements.txt

# A2S uses PyTorch SDPA; FlashAttention and the legacy MIDI stack are optional.
COPY run.sh /usr/local/bin/train-a2s-quartets
RUN chmod 0755 /usr/local/bin/train-a2s-quartets

WORKDIR /workspace/efficient-seq2seq-piano-trans
CMD ["bash", "/usr/local/bin/train-a2s-quartets"]
