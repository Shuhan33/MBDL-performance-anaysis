FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/project
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install numpy matplotlib \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY train_experiment.py .
ENTRYPOINT ["python", "train_experiment.py"]
CMD ["--output-dir", "/work/results"]
