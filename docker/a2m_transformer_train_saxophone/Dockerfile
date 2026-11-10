FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/a2m-matplotlib \
    NUMBA_CACHE_DIR=/tmp/a2m-numba

RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/a2m-requirements.txt
RUN python -m pip install --no-cache-dir \
        torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && \
    python -m pip install --no-cache-dir -r /tmp/a2m-requirements.txt && \
    python -m pip check
# The model uses its existing PyTorch attention fallback, without flash-attn/xformers.
COPY run.sh /usr/local/bin/train-a2m
RUN chmod 0755 /usr/local/bin/train-a2m
WORKDIR /workspace/efficient-seq2seq-piano-trans
CMD ["bash", "/usr/local/bin/train-a2m"]
