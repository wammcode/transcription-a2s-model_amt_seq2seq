# Audio to Score: Quartets con Docker y Slurm

Desde la raíz del proyecto:

```bash
sbatch docker/a2s_transformer_train_quartets/p1.slurm
```

El trabajo solicita una A100 completa de **40 GB** en `dgx`, 6 CPU,
32 GB de RAM del nodo y un máximo de 48 horas. El usuario `wilson` pertenece
al grupo autorizado `dgx-user`. Slurm decide cuándo empieza según los
recursos y la prioridad; el script no interviene en otros trabajos.
`--gres=gpu:a100-40gb:1` reserva la GPU y el contenedor comprueba además
que sólo ve una GPU CUDA con al menos 20 GiB de VRAM.

Se construye la imagen en el nodo asignado con PyTorch/torchaudio 2.7.1 y
CUDA 12.6. No hace falta Docker en el nodo de acceso. El nodo de cómputo
necesita Docker, NVIDIA Container Toolkit y acceso a los registros de imágenes
y paquetes. A2S usa SDPA; no necesita FlashAttention ni la pila MIDI.

El flujo usa `score_quartets.yaml` de esta carpeta, combinado automáticamente
con `efficient-seq2seq-piano-trans/config/score.yaml`:

1. Importa las particiones originales de
   `data/data_quartets/Quartets/a2s_partitions/quartets/{train,val,test}.txt`.
2. Empareja los archivos completos de `Quartets/wav` con los de
   `Quartets/krn_clean`, como polifonía fija (`category: fixed`). Conserva
   cada fragmento y la partición indicada en los archivos originales.
3. Aprende un vocabulario nuevo exclusivamente de train, valida los
   **38.051 pares** (24.351 train, 6.089 validación y 7.611 test) y cachea
   los espectrogramas en disco. No recorta audio ni partituras.
4. Entrena con `--task audio_to_score` **desde cero**, con inicialización
   aleatoria, sin `--resume` ni pesos preentrenados. El encoder se entrena
   también. La salida es una secuencia **kern.

Algunas referencias originales contienen inconsistencias de sincronización
rítmica u omiten duraciones explícitas. Este perfil usa
`data.strict_score_rhythm: false`: conserva sus etiquetas sin corregirlas ni
descartar pares, comprueba estructura, notación y lectura/renderizado con
Verovio, y registra cada incidencia en `prepared.jsonl.validation.json`
(`structural_rhythm_valid: false`, `rhythm_warnings`). El resumen de preparación
cuenta los pares afectados por partición. Esta opción sólo afecta a las
referencias del perfil Quartets: la validación de las predicciones generadas
sigue siendo estricta. Los demás perfiles conservan la validación original.

El modelo conserva 6 capas en encoder y decoder, dimensión 512 y 8 cabezas.
El perfil usa hasta 4 pares por microbatch, acumulación de 4 microbatches,
presupuestos por longitud, recomputación de activaciones y precisión mixta
automática (BF16 en A100). Entrena hasta 10.000 pasos, valida y guarda cada
500, selecciona `best.pt` por pérdida de validación y admite parada temprana
tras 20 validaciones sin mejora. El intervalo de 500 reduce el coste de
recorrer los 6.089 pares de validación. Test se comprueba en preparación;
no interviene en el ajuste del vocabulario ni en la selección del modelo.

Cada envío crea un directorio nuevo, sin reutilizar ni sobrescribir ejecuciones:

```text
efficient-seq2seq-piano-trans/outputs/quartets/slurm-JOBID/
  p1.slurm
  Dockerfile
  requirements.txt
  run.sh
  score_quartets.yaml
  docker-image-id.txt
  manifest.jsonl
  vocab.json
  prepared.jsonl
  prepared.jsonl.validation.json
  prepared.jsonl.tokenizer.json
  features/
  run/config.json
  run/training.jsonl
  run/best.pt
  run/latest.pt
  complete.json
```

Los datos y el código se montan en modo lectura. Sólo la GPU asignada por
Slurm se expone al contenedor. Los artefactos se escriben en el host y
`complete.json` sólo se crea al finalizar correctamente. El lanzador rechaza
argumentos adicionales y una carpeta `run` existente para asegurar un inicio
nuevo en cada envío.

Para consultar estado y registros (reemplazar `JOBID` por el ID del envío):

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList
tail -f docker/a2s_transformer_train_quartets/a2s-quartets-seq2seq-JOBID.out
tail -f docker/a2s_transformer_train_quartets/a2s-quartets-seq2seq-JOBID.err
```

Si se mueve el proyecto, actualizar las rutas absolutas de `--chdir`,
`--output` y `--error` en `p1.slurm` antes de enviarlo.

Documentación del modelo: [Audio to Score](../../efficient-seq2seq-piano-trans/docs/audio_to_score.md).

## Predicciones de test del entrenamiento 106715

```bash
sbatch docker/a2s_transformer_train_quartets/predict_test.slurm
```

Este trabajo carga `outputs/quartets/slurm-106715/run/best.pt`, seleccionado
por pérdida de validación, y genera exclusivamente los 7.611 pares de la
partición original de test. Usa una A100 en `dgx`, reutiliza la imagen y los
espectrogramas del entrenamiento y no modifica los checkpoints.

La generación es autorregresiva greedy, en FP32, con hasta 32 audios por lote
(32.000 frames de presupuesto) y el límite de 16.384 tokens del checkpoint.
Las longitudes de audio controlan los lotes y las máscaras; las referencias
sólo se utilizan después para calcular las métricas. Antes de predecir se
comprueba que el manifiesto coincide exactamente con `test.txt` y se ejecutan
pruebas de generación por lotes y conservación de salidas inválidas.

Los resultados se guardan en:

```text
outputs/quartets/slurm-106715/predictions/test/slurm-JOBID/
  prediction_config.json       # checkpoint SHA256, GPU y opciones
  test_manifest.jsonl          # sólo test, con los pares originales
  results/raw/ID.krn           # todas las salidas sin modificar ni validar
  results/ID.krn               # sólo las que pasan validación estricta
  results/ID.krn.invalid.json  # diagnóstico y tokens si la salida es inválida
  results/ID.report.json       # audio/referencia, validez y errores
  results/examples.jsonl      # índice incremental de todas las predicciones
  results/progress.json       # completadas/total y tiempo
  results/metrics.json        # métricas finales, incluyendo salidas inválidas
  complete.json               # sólo después de completar toda la partición
```

`ID` es el identificador del manifiesto; los informes conservan los nombres
y rutas del WAV y de la referencia KRN correspondientes. `raw/` incluye las
predicciones truncadas o inválidas para no perder ninguna salida ni presentarla
como una partitura validada. Cada envío crea una carpeta independiente.

```bash
squeue -j JOBID
tail -f docker/a2s_transformer_train_quartets/a2s-quartets-test-JOBID.out
```
