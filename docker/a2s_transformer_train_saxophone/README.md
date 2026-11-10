# Audio to Score: Saxophone con Docker y Slurm

Desde la raíz del proyecto:

```bash
sbatch docker/a2s_transformer_train_saxophone/p1.slurm
```

El trabajo solicita una A100 de **40 GB** en `dgx`, 6 CPU, 32 GB de RAM
y un máximo de 48 horas (el límite de esta partición). La RAM de `--mem`
es memoria del nodo; la VRAM se garantiza solicitando la GPU completa con
`--gres=gpu:a100-40gb:1`. El contenedor comprueba además que dispone de
al menos 20 GiB de VRAM. El inicio depende de la disponibilidad de Slurm.

Se construye la imagen en el nodo asignado con PyTorch/torchaudio 2.7.1 y
CUDA 12.6. No hace falta Docker en el nodo de acceso. El nodo de cómputo
necesita Docker, NVIDIA Container Toolkit y acceso a los registros de imágenes
y paquetes. No se instala la pila MIDI ni FlashAttention: A2S admite SDPA.

El flujo usa `efficient-seq2seq-piano-trans/config/score_saxophone.yaml`, que
se combina automáticamente con `config/score.yaml`:

1. Importa las particiones originales `train.txt`, `val.txt` y `test.txt` de
   `data/data_saxophone/real_a2s_sax_dataset/a2s_partitions`.
2. Empareja los WAV completos de `wav` con los KRN de `krn_clean`.
3. Aprende el vocabulario exclusivamente de train, valida los 513 pares
   (360 train, 51 validación y 102 test) y cachea los espectrogramas.
4. Entrena con `--task audio_to_score`, **desde cero**, sin `--resume` ni
   pesos preentrenados. La salida es una secuencia **kern; no se predice MIDI.

El modelo conserva 6 capas en encoder y decoder, dimensión 512 y 8 cabezas.
El perfil usa hasta 4 pares por microbatch, acumulación de 4 microbatches,
presupuestos por longitud, recomputación de activaciones y precisión mixta
automática (BF16 en A100). Entrena hasta 10.000 pasos, valida y guarda cada
100, selecciona `best.pt` por pérdida de validación y para tras 20 validaciones
sin mejora. Test se valida en preparación, pero no interviene en la selección.

Cada envío crea un directorio independiente, sin sobrescribir ejecuciones:

```text
efficient-seq2seq-piano-trans/outputs/saxophone/slurm-JOBID/
  manifest.jsonl
  vocab.json
  prepared.jsonl
  prepared.jsonl.validation.json
  features/
  run/config.json
  run/training.jsonl
  run/best.pt
  run/latest.pt
  complete.json
```

Los datos y el código se montan en modo lectura. Los artefactos se escriben
en el host; `complete.json` sólo se crea al finalizar correctamente.
También se conservan los scripts, el perfil y el ID de imagen de la ejecución.

Para consultar estado y registros:

```bash
squeue -j JOBID
tail -f docker/a2s_transformer_train_saxophone/a2s-saxophone-seq2seq-JOBID.out
tail -f docker/a2s_transformer_train_saxophone/a2s-saxophone-seq2seq-JOBID.err
```

Si se mueve el proyecto, actualizar las rutas absolutas de `--chdir`,
`--output` y `--error` en `p1.slurm` antes de enviarlo.

Documentación del modelo: [Audio to Score](../../efficient-seq2seq-piano-trans/docs/audio_to_score.md)
y [Saxophone](../../efficient-seq2seq-piano-trans/docs/saxophone_long_audio.md).
