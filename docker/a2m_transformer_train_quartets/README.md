# Audio to MIDI: Quartets con Docker y Slurm

Desde la raíz del proyecto:

```bash
sbatch docker/a2m_transformer_train_quartets/p1.slurm
```

El trabajo prepara los datos, entrena **desde cero** el modelo original
`audio_to_midi` y ejecuta automáticamente inferencia sobre **todos** los audios
de test con `run/best.pt`. Guarda un `.mid` por WAV, incluso si la predicción
está vacía. `complete.json` sólo aparece después de completar ambas etapas.

## Datos y modelo

- Particiones: `data/data_quartets/Quartets/amt_partitions/quartets/{train,val,test}.txt`.
- Audio: `data/data_quartets/Quartets/wav`.
- Referencias MIDI: `data/data_quartets/Quartets/midi`.
- Tamaños: 24.351 train, 6.089 val, 7.611 test. Se preservan los pares y splits originales;
  la preparación falla ante archivos ausentes, duplicados o solapamientos.

`../a2m_transformer_common/pipeline.py` adapta los corpus al formato original
`audio.h5`, `subset_*.tsv` y `cache/*.midi-notes.tsv`. Mezcla los canales a mono,
remuestrea a 16 kHz y conserva los WAV/MIDI originales en modo lectura.
Los MIDI de varias pistas se leen con Partitura y se juntan para la tarea
original de transcripción de notas. El tokenizer de este modelo representa
altura, onset, note-off y velocidad; no predice programas ni separa instrumentos.

La configuración `midi.yaml` se combina con el perfil original
`config/main_config.yaml`: **T5 V4 HierarchyPool**, 6 capas en encoder y decoder,
dimensión 512 y 8 cabezas, ventanas de 512 frames (10,24 s) y hasta 1.023 tokens.
Se reutilizan `model.T5.Transformer`, `SingleWavDataset`, el tokenizer MIDI,
el frontend Mel y la pérdida `loss.CrossEntropyLoss` del repositorio.
El adaptador usa Lightning para seleccionar el mejor checkpoint por pérdida de
validación y limita la caché a 32 piezas por worker, evitando abrir miles de
handles HDF5 simultáneos. Usa STFT FFT equivalente al frontend original.
No ejecuta la ruta `a2s` ni convierte partituras a MIDI.

El perfil entrena hasta **10.000 actualizaciones**, con microbatch de 4,
acumulación de 4, FP32 y AdamW a 1e-4; valida cada 500 microbatches
y admite parada temprana tras 20 validaciones sin mejora. El límite de
tiempo de entrenamiento es **30 horas**, con parada controlada para dejar
18 horas del trabajo a la inferencia. Se puede ajustar `midi.yaml` antes de
un nuevo envío. Test no interviene en entrenamiento ni selección del checkpoint.

## Recursos y aislamiento

Solicita una **A100 completa de 40 GB** en `dgx`, 6 CPU y 48 GB de RAM,
por un máximo de 48 horas. El usuario `wilson` pertenece a `dgx-user`.
Slurm escoge entre los nodos autorizados según disponibilidad; puede quedar en
cola. No se detiene ni altera ningún trabajo existente. El contenedor sólo ve
la GPU asignada y verifica al menos **20 GiB totales y libres** antes de empezar.

Docker se construye en el nodo de cómputo con PyTorch/torchaudio 2.7.1 y CUDA
12.6. Requiere Docker, NVIDIA Container Toolkit y acceso al registro de imágenes
y a los paquetes. Se usa la atención PyTorch existente; no se compila FlashAttention.
Cada envío guarda sus scripts, configuración y una copia del código utilizado,
y crea una salida nueva. No reutiliza pesos ni sobrescribe ejecuciones anteriores.
Antes de preparar los datos comprueba que el usuario del contenedor puede leer
y arrancar el lanzador; el resultado se guarda en `entrypoint-check.log`.

## Resultados

```text
efficient-seq2seq-piano-trans/outputs/audio_to_midi/quartets/slurm-JOBID/
  job/                          # scripts, configuración y adaptador utilizados
  source/                       # copia del código del modelo utilizado
  source-commit.txt
  source-working-tree.patch
  docker-image-id.txt
  gpu.json
  dataset/original_*.txt         # copia de las particiones originales
  dataset/manifest.jsonl         # pares originales, duraciones y número de notas
  dataset/audio.h5
  dataset/subset_*.tsv
  dataset/cache/
  dataset/complete.json
  run/config.yaml
  run/checkpoints/best.ckpt      # checkpoint Lightning seleccionado por validación
  run/checkpoints/last.ckpt
  run/best.pt                   # pesos del Transformer para inferencia
  run/latest.pt
  run/csv/
  run/tensorboard/
  training_complete.json
  predictions/test/NOMBRE.mid
  predictions/test/examples.jsonl
  predictions/test/prediction_config.json
  predictions/test/progress.json
  predictions/test/complete.json
  complete.json
```

La generación usa greedy con los límites del perfil y recibe sólo audio,
sin condicionar en referencias ni longitudes de las etiquetas. Las grabaciones
largas se recorren completas en ventanas; se juntan los eventos antes de
reconstruir las notas. `examples.jsonl` relaciona cada WAV con su referencia
y predicción y registra los segmentos que alcanzan el límite de tokens.
Estos segmentos conservan lo generado y no se omiten. No se calculan métricas
MIDI en este flujo; quedan guardados los archivos para evaluación posterior.

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList
tail -f docker/a2m_transformer_train_quartets/a2m-quartets-seq2seq-JOBID.out
tail -f docker/a2m_transformer_train_quartets/a2m-quartets-seq2seq-JOBID.err
```

Para **repetir sólo la inferencia** de un entrenamiento completado:

```bash
sbatch docker/a2m_transformer_train_quartets/predict_test.slurm \
  "$PWD/efficient-seq2seq-piano-trans/outputs/audio_to_midi/quartets/slurm-TRAIN_JOBID"
```

Crea una carpeta nueva `predictions/test/slurm-INFERENCE_JOBID/` y reutiliza
el checkpoint, caché y código del entrenamiento en modo lectura.
No hace falta enviar este trabajo para la primera inferencia: `p1.slurm` ya la ejecuta.

Si se mueve el proyecto, actualizar las rutas absolutas de `--chdir`, `--output`
y `--error` en ambos `.slurm`.
