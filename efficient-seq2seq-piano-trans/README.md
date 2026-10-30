# Sequence-to-Sequence Piano Transcription

This is the repository for [Efficient Transformer-Based Piano Transcription With Sparse Attention Mechanisms](https://arxiv.org/abs/2509.09318).

## Audio to MIDI / Audio to Score

The entry points now accept `--task audio_to_midi` (default, original Hydra
interface) or `--task audio_to_score` (direct generation of Humdrum **kern).
The score task trains from scratch using the same acoustic frontend, Transformer
encoder and autoregressive decoder classes. It does not convert predicted MIDI.

See [the implementation, data contract, sources, limitations and commands](docs/audio_to_score.md)
and [the executed verification report](docs/verification.md).

Complete recordings now use variable lengths (120-second configurable guard),
exact attention blocks with context, and one autoregressive score sequence.
For the local Saxophone dataset, see [training and long-audio details](docs/saxophone_long_audio.md):

```bash
python -m a2s.import_data saxophone \
  --root ../data/data_saxophone/real_a2s_sax_dataset --output outputs/saxophone
python prepare.py --task audio_to_score --config config/score_saxophone.yaml \
  --manifest outputs/saxophone/manifest.jsonl --fit-tokenizer outputs/saxophone/vocab.json \
  --cache-dir outputs/saxophone/features --output outputs/saxophone/prepared.jsonl
python train.py --task audio_to_score --config config/score_saxophone.yaml \
  --tokenizer outputs/saxophone/vocab.json --manifest outputs/saxophone/prepared.jsonl \
  --cache-dir outputs/saxophone/features --output outputs/saxophone/run
python inference.py --task audio_to_score --checkpoint outputs/saxophone/run/best.pt \
  --audio path/to/complete_recording.wav --output outputs/complete_score.krn
```

The full-size model retains six encoder and six decoder layers. Length buckets,
gradient accumulation, mixed precision and disk feature caching control training
cost. `score_long.yaml` provides the general large-corpus profile. No MIDI
predictions or MIDI pretrained weights are used. This change does not establish
transcription accuracy on unseen music; the executed learning checks memorize
three real training pairs and synthetic examples.

For a small synthetic check of all three polyphony cases:

```bash
pip install -r requirements-score.txt
python examples/make_score_demo.py
python prepare.py --task audio_to_score --config config/score_smoke.yaml \
  --manifest outputs/score-demo/manifest.jsonl --output outputs/score-demo/prepared.jsonl
python train.py --task audio_to_score --config config/score_smoke.yaml \
  --manifest outputs/score-demo/prepared.jsonl --output outputs/score-overfit
python inference.py --task audio_to_score --checkpoint outputs/score-overfit/latest.pt \
  --audio outputs/score-demo/open.wav --output outputs/score-overfit/open.krn
python evaluate.py --task audio_to_score --checkpoint outputs/score-overfit/latest.pt \
  --manifest outputs/score-demo/prepared.jsonl --split train --output outputs/score-overfit/evaluation
```

This is a toy memorization check, not a trained real-world transcription system.
Omit `--config config/score_smoke.yaml` to use the full six-layer configuration.
`python -m pytest tests/test_score.py -q` tests the score pipeline;
the complete `tests/` suite additionally requires the original MIDI dependencies.
For the legacy entry points and full regression suite in a new environment,
also install `pip install -r requirements-midi-tools.txt` with matching
PyTorch/torchaudio/torchvision builds.

## Environment Requirement

- PyTorch>=2.2
- Python>=3.10

Install PyTorch
```bash
# CUDA 12.4
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Install FlashAttention
```bash
pip install flash-attn==2.7.4.post1
```

- To use [FlashAttention](https://github.com/Dao-AILab/flash-attention), PyTorch>=2.2 is required. 

Install requirements.
```
$ pip install -r requirements.txt
```
## Dataset Preparation

Download MAESTRO v3.0.0 dataset, and unzip to ./dataset/maestro-v3.0.0
```bash
mkdir dataset
cd dataset
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip
cd ../
```

or use soft symbolic link:

```bash
ln -s path/to/maestro-v3.0.0 ./dataset/maestro-v3.0.0
```


Convert audios into h5py file and save to "dataset/maestro-v3.0.0/audio.h5".
```bash
python data/audio_to_h5_MAESTRO.py
```


Convert midi files to tsv and save to "dataset/maestro-v3.0.0/cache/".
```bash
python data/midi_2_performance_tsv_MAESTRO.py
```

### Training



```bash
# use the default training config in config/main_config.yaml
python train.py

# override
config=experiment_T5_V4_HierarchyPool
python train.py --config-name=$config

# customized config
python train.py --config-name=$config\
  training.batch=16\
  training.learning_rate=1e-4\
  training.training_steps=200000\
  model.checkpoint_path="path/to/checkpoint.ckpt"\
  accelerator=gpu\
  devices=[0,1]\

```

- Config files is managed with [hydra](http://hydra.cc/).


## Evaluation

```bash
# results will be saved to "path/to/checkpoint.ckpt_test/" 
checkpoint_path="path/to/checkpoint.ckpt"
echo $checkpoint_path
python evaluate.py \
  model.checkpoint_path="'$checkpoint_path'" \
  accelerator=gpu \
  devices=[0,1]


```


## Inference

Download the checkpoint from release page to ./checkpoints.

```bash
checkpoint_path=checkpoints/T5_V4_steps_200000.ckpt
python inference.py model.checkpoint_path="$checkpoint_path" audio_path="'audio/Franz-Liszt_Liebestraum.mp3'" +midi_path="outputs/Franz-Liszt_Liebestraum.mid"

```
## Acknowledgements

- rlax59us' [MT3-pytorch](https://github.com/rlax59us/MT3-pytorch)
- Official [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)






