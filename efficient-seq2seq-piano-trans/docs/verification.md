# Verificación ejecutada — 30 de octubre de 2026

Las pruebas se ejecutaron en este checkout, en un entorno `.venv` local
(Python 3.11, PyTorch 2.5.1), con CPU y una RTX 4050 de 6 GB. Se reutilizaron
las instalaciones PyTorch/torchaudio del entorno `a2s_transformer` mediante
`venv --system-site-packages`; los paquetes adicionales se instalaron en
`.venv`, sin modificar ese entorno de origen. FlashAttention no estaba
instalado: las pruebas de atención local utilizaron el backend SDPA.

## Ampliación: Saxophone y grabaciones completas

Ver [guía, fuentes y comandos](saxophone_long_audio.md) y
[resultados estructurados](long_audio_metrics.json).

- La batería actual tiene **40 pruebas**: añade bloques de atención y
  gradientes contra referencia densa, extensión sinusoidal, FFT frente a
  convolución, invariancia al padding, caché, presupuestos, vocabulario
  compacto, reanudación con acumulación/scheduler, parada temprana y
  evaluación independiente de la elección entre campos y bytes.
- Preparados los **513 pares locales** de Saxophone: 360/51/102, audios
  completos de 3,106–25,412 s, vocabulario aprendido sólo en train. Todos
  cargan/renderizan con Verovio; se preservan los originales.
- Modelo completo, BF16 y checkpointing: forward/backward/AdamW con tres
  ejemplos sintéticos completos de **48 s**, mono/fija/abierta, 2400 frames,
  445/767/1165 tokens. Pico por ejemplo **517–523 MiB** asignados por PyTorch
  en RTX 4050. No es una cota para todo corpus ni una comparación de velocidad.
- El modelo completo recorre dos pasos del corpus real (cuatro microbatches
  por paso), valida sus 51 pares y guarda `best.pt` y `latest.pt`. Esto sólo
  comprueba el recorrido; no equivale a entrenarlo hasta convergencia.
- Sobreajuste real: tres pares de train (3,106, 11,223 y 25,412 s), 600 pasos
  con modelo reducido, pérdida **6,46608→0,00051761**, los tres KRN exactos y
  válidos en generación libre. CLI de inferencia verificada con 25,412 s.
- Sobreajuste sintético largo: tres pares de **48 s**, 600 pasos con modelo
  reducido, pérdida final **0,00174551**. Genera completos y exactos los tres
  KRN de **48 compases**, incluidos cambios de voces: SER=0, válidos/renderizables.
  Los bloques de cálculo no requieren unir partituras independientes.

Los artefactos locales están en `outputs/saxophone/` y `outputs/score-long/`.
Los sobreajustes son pruebas de memorización sobre train, **sin evidencia de
generalización**. No se usaron datos de test para los pasos de entrenamiento
ni para ajustar vocabularios. El perfil para entrenar de verdad todo el corpus
es `config/score_saxophone.yaml`; ese entrenamiento hasta convergencia queda
a cargo del experimento del usuario.

## Verificación inicial de clips cortos

- Las **24 pruebas iniciales** siguen incluidas en la batería actual.
- Tokenización y serialización exactas de mono, polifonía fija y abierta;
  Unicode y 37 columnas sin ampliar vocabulario.
- Validación y renderizado con **Verovio** de los tres ejemplos y de ejemplos
  adicionales con adición/intercambio/terminación de voces, ligaduras,
  duraciones recíprocas, puntillos y tuplets.
- Rechazo de columnas inconsistentes, fusiones inválidas, notas solapadas,
  nulos sin evento sostenido, acordes de duración desigual, EOS ausente,
  truncamiento, vocabulario incompatible y checkpoints MIDI en A2S.
- Pares de datos: verificación de desplazamiento temporal real, audio estéreo,
  remuestreo, padding, longitudes y fuga entre particiones.
- Entrenamiento con gradientes en encoder y embedding; carga estricta de
  pesos y optimizador. Entrenamiento de cuatro pasos y ejecución de dos pasos
  más reanudación producen **pesos idénticos** en CPU.
- Equivalencia numérica entre logits con teacher forcing y logits con caché KV
  durante 80 posiciones, incluyendo el límite de ventana de 64, para los tres
  casos. Sólo se comparan posiciones no PAD.
- MIDI: tres patrones de notas pasan tokenización, retorno a eventos/notas,
  escritura/lectura `.mid`, gradiente, paso de optimizador, checkpoint raw y
  generación autorregresiva. Se conserva la selección por defecto y Hydra.
- Se ejecutó además el comando original de inferencia MIDI hasta producir un
  `.mid` legible. Para esa prueba de integración se usó un checkpoint diminuto
  con cabeza fijada a EOS, por lo que es una prueba de recorrido/archivo vacío,
  no una prueba de calidad de transcripción MIDI.

## Sobreajuste desde cero

Se ejecutaron los comandos reproducibles del README con
`config/score_smoke.yaml`: 2 capas, dimensión 64, 2 cabezas, frontend acústico
original de 512 filtros, ventanas locales y pooling `[2,1]`; lote de tres
audios sintéticos de dos segundos. La semilla fue 1234, AdamW lr=0,002,
600 pasos, sin pesos preentrenados.

| Caso | Tokens con EOS | Voces máximas | Generación exacta | **kern válido/renderizable | SER |
|---|---:|---:|---|---|---:|
| Monofónico | 63 | 1 | Sí | Sí | 0 |
| Polifonía fija | 128 | 2 | Sí | Sí | 0 |
| Polifonía abierta | 105 | 3 | Sí | Sí | 0 |

La pérdida de entrenamiento bajó de **5,62003 a 0,00071365**. La generación
libre coincide exactamente con los tres documentos tanto individualmente
como en lote con EOS en longitudes diferentes. El bucle de 600 pasos tardó
aproximadamente 6,8 segundos en esa GPU (no incluye preparación, imports,
extracción acústica ni evaluación; no es un benchmark de eficiencia).

Artefactos locales reproducibles, ignorados por Git:

- `outputs/score-demo/`: WAV, partituras y manifiestos validados.
- `outputs/score-overfit/latest.pt`: checkpoint con configuración/vocabulario.
- `outputs/score-overfit/training.jsonl`: pérdida y tiempos.
- `outputs/score-overfit/evaluation/`: tres `.krn` y `metrics.json`.
- `outputs/score-overfit/open.krn`: resultado de la CLI de inferencia.

El resumen numérico está también en `docs/score_smoke_metrics.json` para que
forme parte del código entregado.

## Configuración completa en la prueba inicial

Se ejercitó además `config/score.yaml` sin reducir capas ni dimensiones:
**22.857.993 parámetros**, seis capas de encoder y seis de decoder, dimensión
512, entrada de 512 frames. Un forward/backward del ejemplo de polifonía
abierta produjo logits `[1,105,259]`, pérdida finita 5,72894 y norma del
gradiente de la proyección del encoder 3,15560. Memoria máxima asignada por
PyTorch en ese proceso: aproximadamente 269 MiB. No se hizo sobreajuste ni
entrenamiento de corpus con el modelo completo.

## Qué demuestra y qué falta

Las pruebas demuestran que la representación conserva la estructura musical,
el pipeline funciona y la misma familia de arquitectura puede aprender los
ejemplos sintéticos y reales. **No demuestran generalización**: el SER=0
corresponde al conjunto usado para el sobreajuste.
No se evaluó MAESTRO ni un corpus A2S independiente, ni se midió la velocidad
del backend FlashAttention. Para una comparación científica hay que igualar
particiones, audio, atención y presupuesto de entrenamiento, y evaluar música
no vista. Las restricciones de validación y segmentación están descritas en
`audio_to_score.md`.

Persisten avisos preexistentes de filtros mel vacíos para 512 filtros/FFT 2048
y de `_self_` en presets antiguos de Hydra, además de deprecaciones de algunas
dependencias. No se cambió el frontend para eliminarlos porque su conservación
es parte del experimento.
