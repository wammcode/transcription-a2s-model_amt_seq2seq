# Comandos básicos de Git

Ejecuta los comandos desde la raíz de este proyecto. La rama principal es `main`.

## Consultar los cambios locales

```bash
git status
```

## Guardar y subir cambios a GitHub

```bash
git add -A
git commit -m "Describe los cambios"
git push origin main
```

Si no hay cambios, no es necesario crear un commit. `git add -A` incluye también los archivos eliminados.

## Descargar los últimos cambios

```bash
git pull --ff-only origin main
```

Si tienes cambios locales que impiden actualizar, guárdalos primero en un commit o con `git stash`; después de actualizar puedes recuperarlos con `git stash pop`. Si las ramas han divergido, revisa el historial antes de integrarlas.

## Descargar el repositorio en otro equipo

```bash
git clone https://github.com/wammcode/transcription-a2s-model_amt_seq2seq.git
cd transcription-a2s-model_amt_seq2seq
```

Para subir cambios necesitas acceso de escritura a GitHub. En este equipo se ha configurado el envío mediante SSH; en otro equipo configura tu autenticación de GitHub antes de hacer `git push`.
