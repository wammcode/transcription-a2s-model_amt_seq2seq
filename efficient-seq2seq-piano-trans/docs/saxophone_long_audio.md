# Saxophone y audio completo

La adaptación usa el dataset local `../data/data_saxophone/real_a2s_sax_dataset`.
Se conservan sus 360 pares de entrenamiento, 51 de validación y 102 de test,
tomados de `a2s_partitions`. Cada WAV completo se empareja con su `krn_clean`:
513 pares, entre 3,106 y 25,412 segundos. Los originales no se modifican.
Se predicen exclusivamente columnas `**kern`, sin `**dynam`.

## Cómo se resuelve la duración

Antes se exigían clips de 10,24 s. Ahora se aceptan longitudes variables,
con una protección configurable de 120 s. Se mantiene un par completo y una
única secuencia autorregresiva; la salida es un `.krn` completo.

La atención local se calcula en bloques de 128 consultas más el contexto de
la ventana original. No se separan notas, compases ni voces en esos límites:
son bloques de cálculo, no ejemplos musicales independientes. Cada capa
conserva sus conexiones originales. Las pruebas comparan resultados y
gradientes con la atención densa, incluyendo padding. La atención cruzada
sigue accediendo a todo el audio mediante el pooling original.

Se conservan log-mel, proyección acústica, seis capas de encoder, seis de
decoder, dimensiones, residuales, MLP y ventanas locales. Los cambios fuera
de representación/salida son de ejecución: longitudes y máscaras, bloques
de atención, pooling parcial sin padding, recomputación de activaciones y
extensión de la misma función sinusoidal. Son necesarios porque cambiar
tokens no resuelve el padding acústico ni las matrices de atención largas.

No hace falta alineación de compases para un par WAV–KRN completo. Si se
quiere dividir una obra externamente, sí hace falta alineación musical fiable
y conservar el estado de voces, notas sostenidas y ligaduras. No se añade
un ensamblador de fragmentos independientes ni streaming de duración ilimitada.

## Fuentes y elección

- El [código oficial de Saxophone](https://github.com/mariaalfaroc/real-a2s/blob/master/my_utils/preprocessing.py)
  calcula espectrogramas de anchura variable, cachea el preprocesamiento y
  rellena hasta el máximo de cada lote. Su [generador](https://github.com/mariaalfaroc/real-a2s/blob/master/my_utils/generators.py)
  lee pares de archivos desde las particiones. Adoptamos ese manejo de pares
  completos y longitudes; conservamos nuestro frontend y decoder.
- El [A2S Transformer de ICASSP 2024](https://github.com/mariaalfaroc/a2s-transformer/blob/main/my_utils/ar_dataset.py)
  usa datos autorregresivos con SOS/EOS, preparación de lotes y evaluación
  individual. Respaldan una secuencia simbólica completa por par; no se copia
  su encoder CNN ni su posición 2D.
- [TUTTI, §4.1.3](https://arxiv.org/html/2609.00640v1#S4.SS1.SSS3)
  describe alineación DualDTW y segmentación por compases con un máximo de
  14,8 s. Es evidencia de que dividir ejemplos musicales requiere alineación;
  no justifica cortar el WAV por segundos y repartir líneas de KRN. Aquí se
  evita esa necesidad para Saxophone manteniendo cada par completo. No se
  incorpora su arquitectura ni su pipeline de síntesis.
- La [descripción oficial del corpus](https://grfia.dlsi.ua.es/audio-to-score/)
  distingue saxofones transpositores. Este importador conserva las etiquetas
  locales sin cambiar alturas: la convención escrita/de concierto debe ser
  consistente al preparar futuras variantes del corpus.

La elección de bloques de atención equivalentes, en lugar de segmentación
musical automática, es una decisión de esta implementación para conservar
la arquitectura solicitada. Fuentes consultadas el 30 de octubre de 2026.

## Entrenar el corpus local

Desde la raíz del repositorio, con `requirements-score.txt` instalado
(en este workspace: `source ../.venv/bin/activate`):

```bash
python -m a2s.import_data saxophone \
  --root ../data/data_saxophone/real_a2s_sax_dataset --output outputs/saxophone

python prepare.py --task audio_to_score --config config/score_saxophone.yaml \
  --manifest outputs/saxophone/manifest.jsonl \
  --fit-tokenizer outputs/saxophone/vocab.json \
  --cache-dir outputs/saxophone/features --output outputs/saxophone/prepared.jsonl

python train.py --task audio_to_score --config config/score_saxophone.yaml \
  --tokenizer outputs/saxophone/vocab.json --manifest outputs/saxophone/prepared.jsonl \
  --cache-dir outputs/saxophone/features --output outputs/saxophone/run

python inference.py --task audio_to_score --checkpoint outputs/saxophone/run/best.pt \
  --audio grabacion_completa.wav --output outputs/partitura_completa.krn

python evaluate.py --task audio_to_score --checkpoint outputs/saxophone/run/best.pt \
  --manifest outputs/saxophone/prepared.jsonl --split test --output outputs/saxophone/test

python train.py --task audio_to_score --resume outputs/saxophone/run/latest.pt \
  --manifest outputs/saxophone/prepared.jsonl --cache-dir outputs/saxophone/features \
  --steps 10000 --output outputs/saxophone/run
```

Los manifiestos, vocabulario y features de esta copia local ya están preparados
en `outputs/saxophone/`; se puede comenzar por el comando de entrenamiento.
`run/best.pt` se creará durante ese entrenamiento; los checkpoints `smoke` y
`full-size-check` son artefactos de verificación, no modelos generales entrenados.

El vocabulario compacto aprende campos frecuentes sólo de train y mantiene
bytes de respaldo. En esta copia: 619 IDs, 31.143 tokens frente a 64.699 con
bytes puros para los 513 documentos (52% menos). Es reversible y conserva
separadores, acordes y cambios de voces; no impone un máximo de polifonía.
La coincidencia exacta en evaluación compara el documento decodificado, ya
que dos secuencias de IDs pueden expresar el mismo texto.

El perfil local usa hasta cuatro ejemplos por microbatch, acumulación de
cuatro microbatches, presupuestos por frames/tokens, BF16 en GPU compatible
(FP16 con escalado en otras CUDA; FP32 en CPU), warmup y decaimiento coseno.
Normaliza la pérdida por el total de tokens válidos acumulados. Guarda estado
del muestreador, RNG, optimizador, scheduler y scaler para reanudar.

Son como máximo 10.000 actualizaciones, validación cada 100, `best.pt` por
pérdida de validación y parada tras 20 validaciones sin mejora. Son valores
iniciales, no hiperparámetros demostrados óptimos. La selección usa pérdida
con teacher forcing; la evaluación final usa generación libre y SER. No se
debe ajustar hiperparámetros usando test. Para corpus grandes generales está
`config/score_long.yaml`, con un presupuesto de entrenamiento mayor.

La caché guarda espectrogramas por ejemplo en disco; no carga el corpus en
GPU/RAM. Sus claves incluyen ruta, tamaño/mtime, intervalo y configuración.
El backend FFT y la convolución original calculan la misma STFT/Hann/mel,
con tolerancia numérica comprobada. Para replicar el backend original se
puede pasar `--set data.stft_backend=conv` en preparación y entrenamiento.

Si falta memoria, reducir `training.batch` y `training.max_batch_frames`,
aumentando acumulación para mantener el lote efectivo. Los overrides se
pasan con `--set clave=valor`. Antes de entrenar, aumentar explícitamente
`data.max_audio_seconds`, `data.max_token_length`, `model.max_decode_length`
y los presupuestos del lote cuando sea necesario. No se truncan etiquetas.
En inferencia se utiliza la configuración guardada en el checkpoint.

## Verificación reproducible y alcance

```bash
python -m pytest tests -q
python examples/make_score_demo.py --output outputs/score-long --repetitions 24
python -m examples.benchmark_score_long --manifest outputs/score-long/manifest.jsonl

python prepare.py --task audio_to_score --config config/score_long_smoke.yaml \
  --manifest outputs/score-long/manifest.jsonl --fit-tokenizer outputs/score-long/vocab.json \
  --cache-dir outputs/score-long/features --output outputs/score-long/prepared.jsonl
python train.py --task audio_to_score --config config/score_long_smoke.yaml \
  --tokenizer outputs/score-long/vocab.json --manifest outputs/score-long/prepared.jsonl \
  --cache-dir outputs/score-long/features --output outputs/score-long/smoke
python evaluate.py --task audio_to_score --checkpoint outputs/score-long/smoke/latest.pt \
  --manifest outputs/score-long/prepared.jsonl --split train --output outputs/score-long/evaluation

python -m examples.make_saxophone_smoke
python train.py --task audio_to_score --config outputs/saxophone/smoke/config.json \
  --manifest outputs/saxophone/smoke/manifest.jsonl --cache-dir outputs/saxophone/features \
  --output outputs/saxophone/smoke/run
python evaluate.py --task audio_to_score --checkpoint outputs/saxophone/smoke/run/latest.pt \
  --manifest outputs/saxophone/smoke/manifest.jsonl --split train \
  --output outputs/saxophone/smoke/evaluation
```

Los 513 pares pasan preparación y carga/renderizado externo. Se aceptan
también los silencios `rr2`, `rr4`, `rr8` de seis partituras, conservando su
texto; el [doble `r` está definido en **kern](https://www.humdrum.org/rep/kern.html).
El lector externo emitió dos avisos de duración de silencios durante la
preparación. Renderizar no certifica toda la semántica métrica o de ligaduras.

Los tres ejemplos sintéticos largos duran 48 s y abarcan mono, fija y abierta
(1→2→3→2→1 voces, repetido). Con el modelo completo, un paso de entrenamiento
por ejemplo, BF16, checkpointing y RTX 4050 de 6 GB consumió un pico de
517–523 MiB asignados por PyTorch. Esto no incluye la reserva del driver, no
es una cota para cualquier música y no demuestra velocidad comparativa.

Con `score_long_smoke.yaml` (dos capas/dimensión 64), 600 pasos sobre esos
tres ejemplos de 48 s dieron pérdida final 0,00174551 y generación libre
exacta en los tres casos: un KRN válido de 48 compases por audio, SER=0.
Es una prueba de memorización, no un resultado sobre un conjunto independiente.

El sobreajuste real usa exclusivamente tres pares de train (3,106, 11,223 y
25,412 s), modelo reducido de dos capas/dimensión 64, 600 pasos y semilla 1234:
pérdida 6,46608→0,00051761; tres documentos exactos en generación libre, SER=0.
También se verificó la CLI completa con el audio real de 25,412 s, sin límites
manuales. El modelo completo recorrió dos actualizaciones del corpus local y
validación de sus 51 ejemplos; no se entrenó todavía hasta convergencia.

Estos resultados demuestran funcionamiento y capacidad de memorizar los
ejemplos. La calidad sobre música no vista y la extrapolación a duraciones
mayores que las de entrenamiento siguen sin demostrarse. La atención cruzada
mantiene coste frames × tokens; no se promete duración ilimitada. La
generación greedy puede producir KRN inválido: se rechaza con diagnóstico,
sin inventar notas o reparar una partitura silenciosamente.
