# Documentación EMOTRIX

El proyecto Emotrix es una plataforma que permitirá el estudio, identificación y manipulación de emociones humanas. La plataforma se podrá utilizar para fines educativos, médicos o recreacionales, entre otros. A la fecha, existen distintos dispositivos que permiten imitar inteligencia emocional, los cuales tienen precios elevados. Es por esto por lo que la plataforma tiene un enfoque de código libre para permitir el desarrollo de forma abierta y evitar cualquier tipo de restricciones a los usuarios. El ámbito al que pertenece el proyecto es a la interacción humano – computador (HCI). Este proyecto ha sido desarrollado desde 2016 por estudiantes de la Universidad del Valle de Guatemala.

**IMPORTANTE** todo lo desarrollado por EMOTRIX Fase II se encuentra dentro de los directorios: 
> /emotrix/2017/... 

> /R/... 

Lo demás fue desarrollado por EMOTRIX Fase I.
## Requisitos Previos
1. Python 2.7
2. pip
3. R
4. RStudio
5. Emotiv EPOC headset + Emotiv Xavier ControlPanel

## Instalación
```
pip install requirements.txt
```

## Ejecutar el SDK
```
python main.py
```
## API

### Estímulos Visuales
Nombre del archivo: [main.py](https://github.com/emotrix/Emotrix/blob/master/emotrix/2017/Estimulos_Visuales/main.py)

Directorio: */emotrix/2017/Estimulos_Visuales/...*

Este programa genera un archivo con las lecturas obtenidas del casco Emotiv EPOC y a su vez presenta diferentes estímulos visuales que buscan estimular diferentes emociones en el sujeto que lo esté usando.

### Estímulos Auditivos
Nombre del archivo: [gui.py](https://github.com/emotrix/Emotrix/blob/master/emotrix/2017/auditivos/gui.py)

Directorio: */emotrix/2017/auditivos/...*

Este programa genera un archivo con las lecturas obtenidas del casco Emotiv EPOC y a su vez presenta diferentes estímulos auditivos que buscan estimular diferentes emociones en el sujeto que lo esté usando.

## Código Fuente Para Estados de Relajación
Todo el código fuente para esta sección se encuentra dentro del directorio:
> /R/RelaxState-Scripts/...

### Generación de Gráficas 
Nombre del archivo: [*graphics.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/graphics.R)

Programa que lee cada uno de los archivos generados por el API, luego de la experimentación, y genera gráficas de las lecturas de cada electrodo del Emotiv y de las ondas presentes en cada segundo de experimentación, a una frecuencia de muestreo de 2048 Hz.


### Generar data utilizada para entrenar algoritmos y Validación Cruzada
Nombre del archivo: [*cleanData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/cleanData.R)

Programa que lee cada uno de los archivos generados por el API y genera un archivo por cada uno de los archivos anteriores, filtrando data de únicamente el estado de relajación. 

Nombre del archivo: [*bindData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/BindData.R)

Programa que lee cada uno de los archivos generados por el programa anterior y los une en un mismo data frame. Separa esta data en train y test y a por cada segundo de experimentación asigna un número de observación a cada una, exporta estos DFs con toda la data generada.

Nombre del archivo: [*settingData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/settingData.R)

Programa que lee los DFs generados por el programa anterior, aplica un filtro de banda para agarrar únicamente las frecuencias de las ondas EEG correspondientes y aplica la transformada de wavelets y obtener que onda está presente en cada segundo de la experimentación.

### Regresión Logística
Nombre del archivo: [*lr.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/lr.R)

Programa que lee los archivos generados mediante el programa [*settingData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/settingData.R), entrena el algoritmo de regresión logística, hace la validación cruzada y devuelve el porcentaje de eficiencia.

### Maquinas de Soporte Vectorial (SVM)
Nombre del archivo: [*svm.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/svm.R)
Programa que lee los archivos generados mediante el programa [*settingData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/settingData.R), entrena el algoritmo de maquinas de soporte vectorial, hace la validación cruzada y devuelve el porcentaje de eficiencia.


### K-Medias
Nombre del archivo: [*kmeans.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/kmeans.R)
Programa que lee los archivos generados mediante el programa [*settingData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/settingData.R), entrena el algoritmo de K-Medias, hace la validación cruzada y devuelve el porcentaje de eficiencia.


### Redes Neuronales
Nombre del archivo: [*neuralnet.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/neuralnet.R)

Programa que lee los archivos generados mediante el programa [*settingData.R*](https://github.com/emotrix/Emotrix/blob/master/R/RelaxState-Scripts/settingData.R), entrena el algoritmo de redes neuronales, hace la validación cruzada y devuelve el porcentaje de eficiencia.

## Código Fuente Para Estados Emocionales
Todo el código fuente para esta sección se encuentra dentro del directorio:
> /R/Emotions-Scripts/...


### Obtención de Emociones Seleccionadas
Nombre del archivo: [*SelectedEmotions.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/SelectedEmotions.R)

Programa que lee cada uno de los archivos generados por el API, luego de la experimentación, y genera un archivo por cada uno de los archivos anteriores, con la emoción seleccionada por segundo para cada estimulo.
### Generación de Graficas
Nombre del archivo: [*Graphics.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/Graphics.R)

Programa que lee cada uno de los archivos generados por el API, luego de la experimentación, y genera un graficas de las lecturas de cada electrodo del Emotiv y de las onda presente en cada segundo de experimentación.

### Obtención de Datos de Entrenamiento y Validación Cruzada
Nombre del archivo: [*SetTrainingData.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/SetTrainingData.R)

Programa que lee cada uno de los archivos generados por el API, luego de la experimentación, y genera un archivo con los datos separados para el entrenamiento y validación de los algoritmos.

### Regresión Logística
Nombre del archivo: [*lr.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/lr.R)

Programa que lee los archivos generados mediante el programa *SetTrainingData.R*, entrena el algoritmo de regresión logística, hace la validación cruzada y devuelve el porcentaje de eficiencia.
### Maquinas de Soporte Vectorial (SVM)
Nombre del archivo: [*SVM.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/SVM.R)

Programa que lee los archivos generados mediante el programa *SetTrainingData.R*, entrena el algoritmo de maquinas de soporte vectorial (SVM), hace la validación cruzada y devuelve el porcentaje de eficiencia.
### K-Medias
Nombre del archivo: [*k-means.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/k-means.R)

Programa que lee los archivos generados mediante el programa *SetTrainingData.R*, utiliza el algoritmo de k-medias para la generación de clústeres para la predicción de los estados emocionales. Retorna la eficiencia del algoritmo.
### Redes Neuronales
Nombre del archivo: [*Neural-Networks.R*](https://github.com/emotrix/Emotrix/blob/master/R/Emotions-Scripts/Neural-Networks.R)

Programa que lee los archivos generados mediante el programa *SetTrainingData.R*, utiliza el algoritmo de redes neuronales para la predicción de los estados emocionales. Retorna el porcentaje de eficiencia.
## Recomendaciones
- Utilizar un sistema operativo basado en Linux
- Utilizar Visual Studio Code

## Autores
- Ángel Morales
- Cesar Ruiz
- Diego Jacobs
- Henzer García
- Kevin García
- Mario Barrientos
- Sergio Gómez
