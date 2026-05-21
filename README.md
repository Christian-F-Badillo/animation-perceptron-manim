# Animación de Entrenamiento de Perceptrón Multicapa

En este repositorio se incluye el código para la animación del proceso de entrenamiento de un perceptrón multicapa hecho desde 0 en Python 3.12.
Para la animación se uso la famosa paqueteria [Manim](https://www.manim.community/) para animaciones matemáticas con python.

## Animación Objetivo

[<video src="(https://github.com/user-attachments/assets/63dc0919-e5d7-42dc-80fd-aca1d3dac3b8)" width="100%" controls autoplay loop muted></video>
](https://github.com/user-attachments/assets/63dc0919-e5d7-42dc-80fd-aca1d3dac3b8
)

## Instalación

Para correr la renderización de la animación es necesario clonar el repositorio de Github.

```bash
git clone https://github.com/Christian-F-Badillo/animation-perceptron-manim.git
cd animation-perceptron-manim
```

Si no tienes git instalado puedes descargarlo en su página oficial [Git](https://git-scm.com/install).
Para crear el entorno virutal de python con las dependencias para la animación, se recomienda usar `uv` un gestor de projectos y paquetes para Python, lo puedes instalar siguiendo la guía de su página oficial [uv](https://docs.astral.sh/uv/getting-started/installation/) (se asumirá que se tiene instalado `uv` para los siguientes pasos).

Dentro del directorio del repositorio en tu copia local, ejecuta el siguiente comando para crear el entorno virtual

```bash
uv sync
```

## Render

Puedes renderizar la animación con distinta calidad de video, para una prueba rápida usa el flag `-ql` para renderizar a baja calidad, `-qh` para calidad HD, `-qp` para 2k y `-qk` para 4k, por ejemplo:

```bash
uv run manim -pql main.py VisualizeNeuralNetworkCategorization
```

La flag `-p` sirve para reproducir la animación cuando se renderice. Entre mayor calidad mayor será el tiempo para renderizar y más recursos se necesitan para el proceso.
Por defecto se generan archivos de formato `.mp4`, si prefieres un formato `.gif` utiliza el flag `--format gif`.

```bash
uv run manim -pql --format gif main.py VisualizeNeuralNetworkCategorization
```
