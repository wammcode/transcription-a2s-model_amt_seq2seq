# Transcripción directa de audio a **kern

Se selecciona `--task audio_to_midi` o `--task audio_to_score` en `prepare.py`,
`train.py`, `inference.py` y `evaluate.py`. Sin argumento se ejecuta MIDI.
La nueva rama no usa una predicción MIDI, `music21` ni cuantización de eventos
para fabricar partituras: la distribución del decoder predice directamente los
tokens reversibles de la partitura de referencia. Para Saxophone local y
grabaciones largas, usar la [guía específica](saxophone_long_audio.md).

## Arquitectura conservada y adaptación

| Componente | Implementación |
|---|---|
| Acústica | La clase original `data.mel.MelSpectrogram`: 16 kHz, FFT 2048, hop 320, 512 filtros HTK, 20–7600 Hz, log de magnitud; misma eliminación de la última muestra y transposición. |
| Encoder | `model.Encoder.Encoder`: mismas proyección, posición sinusoidal, atención, normalización, MLP y residuales; ejecución por bloques y padding enmascarado. |
| Decoder | `model.Decoder.Decoder`, mismas capas, posición, atención causal, atención cruzada, normalización, MLP y caché KV. |
| Tamaño predeterminado | 6 capas en encoder y decoder, dimensión 512, 8 cabezas, head_dim 64, MLP 1024, dropout 0,1, como T5 original. |
| Atención eficiente | Se mantienen ventanas locales de 64 y pooling cruzado `[4,4,2,2,1,1]`. |
| Salida | 259 IDs básicos o vocabulario compacto aprendido sólo de train, con respaldo en bytes; embedding y proyecciones dimensionados con `vocab_size`. Se usa la cabeza general existente. Las dos cabezas MIDI permanecen pero no se usan. |
| Posición | Misma fórmula sinusoidal, extensible en el encoder; salida configurable (4096 por defecto, 16384 en perfiles largos); no se añaden posiciones aprendidas ni capas. |
| Optimización A2S | AdamW, teacher forcing con BOS desplazado, entropía cruzada ignorando PAD, recorte de gradiente; todos los pesos se inicializan desde cero. |

**Diferencia respecto al preset MIDI V4:** sus reglas `posición % 3` y el
seguimiento del último token onset dependen de tripletas temporales MIDI. Las
duraciones escritas de **kern no determinan segundos de ejecución bajo rubato.
Aplicar esas reglas a bytes de partitura seleccionaría cabezas y ventanas sin
significado. Se selecciona la ruta general de atención cruzada y salida que ya
existe en el mismo decoder (`hybrid_global_local_cross_attn=false`, ventana
cruzada MIDI=0); el pooling sigue activo. No se reemplaza ninguna capa.

Añadir marcas de tiempo a cada grupo de bytes requeriría supervisión temporal
interna adicional y cambiaría el objetivo simbólico del experimento. Por eso
se utiliza la ruta independiente del tipo de token. Para una comparación con
la misma configuración de atención en ambas tareas se incluye
`experiment_T5_task_comparison.yaml` para MIDI. No presentar una comparación
contra V4 sin controlar esta diferencia como efecto exclusivo de la notación.
Los presets originales V0–V4 y los IDs MIDI se conservan.

Fuera de la representación/salida hay ajustes de ejecución mínimos:
FlashAttention pasa a ser opcional y se ofrece SDPA con la misma máscara local
(permite CPU y GPUs sin esa extensión); se desactiva su dropout durante `eval()`.
Esto no modifica los parámetros ni las conexiones de atención. SDPA divide
las consultas en bloques con el halo de la ventana local original; evita
materializar una máscara local NxN. La atención cruzada sigue siendo global
y su trabajo total crece con frames × tokens. Se añade recomputación de
activaciones, padding enmascarado y pooling que excluye padding del último
grupo. No puede resolverse esto cambiando únicamente tokens: son necesidades
de ejecución de secuencias acústicas variables. FFT calcula la misma STFT
Hann que la convolución original, con pequeñas diferencias de redondeo;
`stft_backend: conv` conserva el backend original. Se quitaron imports
innecesarios y se corrigieron dispositivos
del audio mono y sincronización CUDA en la inferencia MIDI.

## Representación y tres casos de polifonía

`KernTokenizer` reserva PAD=0, BOS=1, EOS=2 y codifica cada byte UTF-8 como
`byte+3`. Las etiquetas contienen el documento y EOS; BOS se incorpora por
desplazamiento en el modelo. No se aprende un vocabulario del conjunto de test,
no hay UNK, y la versión/especificación/hash del tokenizador queda en checkpoint.
Se normalizan únicamente CRLF/CR a LF. Tabs, espacios y saltos de línea tienen
IDs distintos. También se preservan metadatos Unicode, aunque puedan no ser
deducibles del audio; conviene normalizarlos consistentemente en un corpus real.

Se conserva el documento **kern en vez de reducirlo a una lista de alturas:

- Monofonía: una columna, notas y silencios con duración explícita.
- Polifonía fija: varias columnas; tabulador para simultaneidad entre voces,
  espacio para las notas de un acorde, `.` para continuación de una voz.
- Polifonía abierta: se generan `*^`, `*v`, `*+`, `*x` y `*-` como parte de la
  secuencia. El número de voces puede cambiar dentro del compás. Se preservan
  `*staffN`, claves, armaduras, indicadores métricos y barras.

No existe un máximo de voces/pentagramas en el vocabulario o en la red. Sí hay
límites prácticos de longitud/memoria. Se verifican 37 columnas en tokenización
y ejemplos musicales de 1, 2 y 1→2→3→2→1 voces, incluyendo acordes, silencios y
duraciones distintas. Los dígitos permiten duraciones recíprocas no vistas,
puntillos, tuplets y octavas sin enumerar combinaciones. El coste es una
secuencia más larga que con un vocabulario de símbolos musicales completos.

`KernFieldTokenizer` (v2) comprime campos frecuentes completos sin eliminar
separadores ni operaciones de voces. Se ajusta con `prepare --fit-tokenizer`
exclusivamente sobre train. Los campos desconocidos se descomponen en bytes;
no introduce UNK ni un máximo de voces. No combina ni intercambia vocabularios
entre checkpoints: cada uno lleva su especificación completa y hash.

## Datos, alineación y segmentación

El contrato es JSONL, una fila por **par completo** o intervalo ya alineado:

```json
{"id":"obra07_003","piece_id":"obra07","split":"train","category":"open","audio":"audio/obra07.wav","score":"scores/obra07_003.krn","start":12.34,"end":21.81}
```

`start/end` son segundos de ejecución, intervalo `[start,end)`. Las rutas se
resuelven respecto al manifiesto. `piece_id` debe identificar la composición
para mantener todas sus interpretaciones en una única partición. Se comprueba
también que un mismo archivo de audio no aparezca en distintas particiones.
`category` es una etiqueta opcional para informes separados; no limita al modelo.

Cada `.krn` contiene **todo y sólo el intervalo anotado**, con cabecera,
contexto de claves/armadura/metro/pentagramas, barras y terminadores. La
alineación debe proceder de anotaciones revisadas, de un alineador externo o
del calendario exacto de síntesis. No se deduce a partir de un tempo global
cuando hay rubato. No se necesita alineación de cada byte a un frame: la
atención cruzada aprende la correspondencia dentro de cada par.

La configuración actual acepta el audio completo, hasta `max_audio_seconds`
(120 s, configurable). No hace falta cortar los pares de Saxophone. Los
bloques de atención son una partición del cálculo: comparten el contexto
original y producen una única secuencia autorregresiva y un único `.krn`.
Para corpus grandes se recomienda `score_long.yaml`, y para Saxophone local
`score_saxophone.yaml`, con caché en disco, lotes por longitud y acumulación.

También sigue disponible el modo de clips fijos (`variable_length: false`,
duración `n_frames*hop/sample_rate`), usado por la prueba histórica reducida.
Si un corpus se segmenta externamente, debe hacerse en límites musicales
alineados y conservar contexto, notas sostenidas y ligaduras: el código no
inventa alineaciones ni corta un `.krn` por líneas.

`prepare.py` valida límites reales del audio, lectura, mezcla a mono,
remuestreo, score, número de tokens y ausencia de fuga entre particiones.
Rechaza intervalos demasiado largos y no trunca etiquetas. Guarda un
manifiesto resuelto y los informes `.validation.json` y `.tokenizer.json`.
En modo variable sólo se completa el último hop (y el mínimo de STFT para
audios muy cortos). Los espectrogramas se rellenan hasta el máximo del lote,
con máscaras de longitud; los PAD de etiquetas se excluyen de la pérdida.

Los ejemplos incluidos tienen dos compases a negra=240, exactamente dos
segundos. `examples/make_score_demo.py` crea audio por osciladores con el
calendario explícito de notas de cada partitura, sin descargar datos.

## Uso

Instalar un PyTorch apropiado al hardware y después:

```bash
cd efficient-seq2seq-piano-trans
pip install -r requirements-score.txt
python -m pytest tests/test_score.py -q

# Validación del corpus y entrenamiento completo desde cero
python prepare.py --task audio_to_score --config config/score_long.yaml \
  --manifest pares.jsonl --fit-tokenizer vocab.json --cache-dir outputs/features \
  --output preparados.jsonl
python train.py --task audio_to_score --config config/score_long.yaml \
  --tokenizer vocab.json --cache-dir outputs/features \
  --manifest preparados.jsonl --output outputs/a2s

# Reanudar exclusivamente un checkpoint de esta tarea
python train.py --task audio_to_score --manifest preparados.jsonl \
  --resume outputs/a2s/latest.pt --steps 250000 --output outputs/a2s

# Grabación completa; no requiere --start/--end
python inference.py --task audio_to_score --checkpoint outputs/a2s/latest.pt \
  --audio obra.wav --output outputs/obra.krn

# Evaluación libre, sin pasar la referencia al decoder
python evaluate.py --task audio_to_score --checkpoint outputs/a2s/latest.pt \
  --manifest preparados.jsonl --split test --output outputs/a2s-test

# Flujo MIDI original y control con atención igualada
python train.py --task audio_to_midi --config-name=experiment_T5_task_comparison
python inference.py --task audio_to_midi model.checkpoint_path=checkpoints/model.ckpt \
  audio_path=obra.wav +midi_path=outputs/obra.mid
```

Los argumentos A2S son argparse y su YAML se pasa con `--config`; MIDI sigue
usando overrides Hydra. `--device cpu` es válido para A2S. Los comandos del
README ejecutan el sobreajuste con `score_smoke.yaml`, que reduce sólo
dimensión/capas/longitud para la prueba, manteniendo las mismas clases.
Para probar también MIDI en un entorno nuevo, instalar
`requirements-midi-tools.txt` y ejecutar `python -m pytest tests -q`.
PyTorch, torchaudio y torchvision deben ser builds compatibles entre sí.

El checkpoint A2S incluye configuración acústica/modelo, vocabulario, pesos,
optimizador, scheduler, escalado AMP, cursor del muestreador, paso, estados RNG
y huella del manifiesto/etiquetas/metadatos de audio. Se comprueba
la tarea y se carga `strict=True`. Un checkpoint MIDI no se acepta ni como
inicialización ni como resume. La huella incluye tamaño/mtime del audio,
no su contenido binario: mantener inmutable el corpus. Los checkpoints A2S
v1 anteriores se pueden cargar para inferencia; no se garantiza reanudación
exacta con su huella y muestreador antiguos. El checkpoint MIDI conserva su formato
original de `state_dict`.

## Validación, evaluación y límites

Sólo se escribe `.krn` después de comprobar EOS, UTF-8, estructura/rítmica
interna y carga/renderizado con Verovio. Una generación truncada o inválida
deja `.invalid.json` y salida no exitosa en inferencia; no se reparan notas,
voces o compases silenciosamente. Un archivo previo en la misma ruta se
conserva si falla el nuevo intento: consultar siempre el informe del intento.

El validador interno verifica cabeceras, anchura de cada registro, operaciones
de columnas, continuidad temporal, duraciones explícitas, acordes de duración
uniforme y terminación. Verovio ofrece un lector independiente. **No es una
certificación exhaustiva de Humdrum**: no comprueba toda la semántica de
ligaduras, indicaciones métricas, repeticiones ni toda extensión RDF.
El pipeline admite spines `**kern`; rechaza spines auxiliares como `**text` o
`**dynam`, duraciones implícitas y algunos signos extendidos. El tokenizador
sí puede representarlos, pero ampliar el validador es necesario antes de
aceptarlos en datos. No limita la cantidad ni variación de voces.

La inferencia produce una partitura por grabación completa o intervalo
solicitado. No necesita ensamblar los bloques de atención. Los excesos de
audio o tokens son errores explícitos: límites configurables no equivalen a
duración ilimitada ni garantizan precisión fuera de las longitudes entrenadas.
No se implementa streaming de duración ilimitada ni ensamblado de fragmentos
musicales cortados externamente.

`evaluate.py` informa tasa de archivos válidos, coincidencia exacta, distancia
de edición de bytes y SER de símbolos separados por espacios, tabs y filas,
global y por categoría. Incluye las predicciones inválidas, no sólo los éxitos.
No es un F1 de notas ni una métrica invariante a notaciones equivalentes o
permutaciones de voces. Para calidad musical real, añadir evaluación de
notas/duraciones/voces alineadas en un corpus independiente; no comparar el
SER directamente con el F1 MIDI como si fueran la misma magnitud.

El entrenamiento A2S es de un dispositivo, con guardado periódico, selección
por pérdida de validación y parada temprana opcional. La caché persistente de
features se activa con `--cache-dir`; la caché RAM se reserva para pruebas
pequeñas. No se implementó DDP. La arquitectura completa se ha ejercitado;
el sobreajuste de tres grabaciones reales no demuestra generalización.

## Fuentes y decisión de diseño

1. Wei y Yoshii, *Efficient Transformer-Based Piano Transcription With Sparse
   Attention Mechanisms* (2025), secciones III–IV, [paper](https://arxiv.org/abs/2509.09318)
   y [repositorio original](https://github.com/WX-Wei/efficient-seq2seq-piano-trans).
   Se consultó también `../../paper/efficient-seq2seq-piano-trans.pdf`.
   Motiva conservar log-mel, Transformer, ventanas locales y pooling. Se usa
   el código como referencia reproducible: el vocabulario MIDI del checkout
   tiene algunas constantes distintas de las enumeradas en el texto del paper.
2. Alfaro-Contreras et al., *A Transformer Approach for Polyphonic Audio-to-Score
   Transcription*, ICASSP 2024, §§2.1–2.2,
   [DOI](https://doi.org/10.1109/ICASSP48485.2024.10447162),
   [código oficial](https://github.com/mariaalfaroc/a2s-transformer).
   Se leyó `../../paper/Audio-to-Score_Transcription.pdf`. Su serialización de
   columnas/filas de Kern y entrenamiento autorregresivo respaldan esta salida
   directa. Aquí se conservan los separadores literales y se descomponen los
   símbolos en bytes. No se adopta su CNN ni su posición 2D, porque cambiarían
   la arquitectura acústica de este experimento.
3. [vicarmar/audio2score](https://github.com/vicarmar/audio2score), README y
   preparación de datos: contempla codificaciones `single_ext` y polifonía
   `unconstrained`; refuerza evitar reducciones a un número fijo de voces.
   No se incorpora su arquitectura DeepSpeech2.
4. [SheetSage2Kern / HookKern](https://github.com/Multimodal-Music-Research-Lab/SheetSage2Kern_model),
   [paper 2026](https://arxiv.org/abs/2608.06165): referencia adicional de salida
   directa **kern, datos emparejados y métricas de símbolos. Sus variantes CNN
   y MuQ son alternativas arquitectónicas que no se trasladan aquí.
5. [Especificación **kern](https://www.humdrum.org/rep/kern/) y documentación
   de [voces variables](https://doc.verovio.humdrum.org/humdrum/getting_started/multiple_voices.html)
   y [pentagramas](https://doc.verovio.humdrum.org/humdrum/getting_started/multiple_staves.html):
   fundamentan duración recíproca, nulos, acordes, separadores y cambios de
   columnas. La decisión de usar bytes reversibles y validación posterior es
   de esta implementación, no un resultado experimental de esos trabajos.

Fuentes consultadas el 30 de octubre de 2026. No se descargaron pesos ni corpus
de los proyectos citados.
