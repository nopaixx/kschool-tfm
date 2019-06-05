# CAR DRAWING BRUSH TOOL!

Proyecto Final de Master
El objetivo de este proyecto es presentar una herramienta de dibujo que mediente herramientas de deeplearning permita el dibujo de un coche finalizando partido de los trazos a mano alzada de los mismo.


![Portada](resources/portada.png)


# Preparación del entorno

Pre-requistio:

1.-Cuenta de Kaggle

2.-Descarga 3 ficheros .zip en la carpeta /downloads/

3.-PC >=16Gigas de RAM

4.-PC recomendable con GPU

5.-Tiempo aprox descarga dataset 4 horas

6.-Tiempo aprox train de modelos 24-48 horas


Fichero1(2GB).-[StandordCars DataSet](https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder)

Fichero2(22GB).-[Carvana cars Image Mask](https://www.kaggle.com/c/carvana-image-masking-challenge)

Fichero3(500MB).-[Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)


    git clone -b master https://github.com/nopaixx/kschool-tfm.git

 1. Install VirtualEnv `pip install virtualenv`
 2. Create virtual env with enviroments `virtualenv -p python3 venv`
 3. Activate virtualenv `source venv/bin/activate`
 4. Install requirements `pip install -r requirements.txt`
 5. Check python version >3.5 `python --version`
 6. Prepare intel image classification dataset 

`unzip downloads/intel-image-classification.zip -d input/intel-image-classification`

`sudo unzip input/intel-image-classification/seg_train.zip -d input/intel-image-classification/seg_train/`

 7. Prepare carvana image masking dataset 
`unzip downloads/carvana-image-masking-challenge.zip -d input/carvana-image-masking-challenge sudo unzip input/carvana-image-masking-challenge/train.zip -d input/carvana-image-masking-challenge/ sudo unzip input/carvana-image-masking-challenge/train_masks.zip -d input/carvana-image-masking-challenge/`
 8. Prepare standford cars dataset `unzip downloads/stanford-car-dataset-by-classes-folder.zip -d input/stanford-car-dataset-by-classes-folder sudo unzip input/stanford-car-dataset-by-classes-folder/car_data.zip -d input/stanford-car-dataset-by-classes-folder/car_data`


## Descripción de ficheros

 - **Carvana car Image Mask**.- Este dataset contiene imagenes de coches con su mascara. Este dataset se usa para entrenar un modelo previo para extraer la mascara del coche del fondo de la imagen con el objetivo de mejorar el algoritmo de Canny Edge detection
 - **Intel image classification**.- Contiene imagenes de fondos (ciudades, paisajes etc...) este dataset se usa para mejorar la augmentación de datos del primer Carvana car Image Mask
 - **StandFord cars dataset**.- Este dataset contiene mas de 15 mil imagenes de coches, este datesaet es usado en el último modelo para regenear los coches a partir de los trazos


## Modelo 1 .- Extraer Mascara del coche
El ojetivo de este modelo es usar el Dataset de Carvana Image con el objetivo de entrenar una red neuronal capaz de extraer la mascara de un coche. Una vez obtenida esta mascara la aplicaremos al dataset StandFordCard para aplicar al coche (y solo al coche) el algoritmo de canny edgedetection que conformara el input para la siguiente red neuronal capaz de generar (o regenerar) un coche en funcion de sus trazos.

Problematicas detectadas y puntos fuertes:

El dataset de carvana, a pesar de tener aproximadamente 4000 imagenes de train esto es solo 400 coches únicos ya que los coches se repiten desde diferente angulo y siempre con el mismo fondo, esto es son solo  4000 images incluida la augmentación de datos.

Estos dos echos juntos (mismo fondo y pocos modelos de coches) en las primeras versiones hizo que el modelo una vez entrenado y aplicado en cualquier coche fuera del dataset de carvana (como Standford cars) produciendo resultados muy malos.

La inovación en este punto llego cuando se incorpora una augmentación custom, donde se incorpora el dataset Intel Image Classification, gracias a la gran variedad de fondos disponibles en este dataset y aplicand técnicas de extracción de maskara y interpolación de una imagen encima de otra se consiguio mejorar mucho el model.

Pasos Para la augmentacion:

 1. Extracción de un coche mediante su mascara
 2. Fusionar un coche dentro de un fondo aleatorio


![Augmentacion](resources/augmentacion.png)

 

## Model 2.-
## Rename a file

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## Delete a file

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

## Export a file

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.


# Synchronization

Synchronization is one of the biggest features of StackEdit. It enables you to synchronize any file in your workspace with other files stored in your **Google Drive**, your **Dropbox** and your **GitHub** accounts. This allows you to keep writing on other devices, collaborate with people you share the file with, integrate easily into your workflow... The synchronization mechanism takes place every minute in the background, downloading, merging, and uploading file modifications.

There are two types of synchronization and they can complement each other:

- The workspace synchronization will sync all your files, folders and settings automatically. This will allow you to fetch your workspace on any other device.
	> To start syncing your workspace, just sign in with Google in the menu.

- The file synchronization will keep one file of the workspace synced with one or multiple files in **Google Drive**, **Dropbox** or **GitHub**.
	> Before starting to sync files, you must link an account in the **Synchronize** sub-menu.

## Open a file

You can open a file from **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Open from**. Once opened in the workspace, any modification in the file will be automatically synced.

## Save a file

You can save any file of the workspace to **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Save on**. Even if a file in the workspace is already synced, you can save it to another location. StackEdit can sync one file with multiple locations and accounts.

## Synchronize a file

Once your file is linked to a synchronized location, StackEdit will periodically synchronize it by downloading/uploading any modification. A merge will be performed if necessary and conflicts will be resolved.

If you just have modified your file and you want to force syncing, click the **Synchronize now** button in the navigation bar.

> **Note:** The **Synchronize now** button is disabled if you have no file to synchronize.

## Manage file synchronization

Since one file can be synced with multiple locations, you can list and manage synchronized locations by clicking **File synchronization** in the **Synchronize** sub-menu. This allows you to list and remove synchronized locations that are linked to your file.


# Publication

Publishing in StackEdit makes it simple for you to publish online your files. Once you're happy with a file, you can publish it to different hosting platforms like **Blogger**, **Dropbox**, **Gist**, **GitHub**, **Google Drive**, **WordPress** and **Zendesk**. With [Handlebars templates](http://handlebarsjs.com/), you have full control over what you export.

> Before starting to publish, you must link an account in the **Publish** sub-menu.

## Publish a File

You can publish your file by opening the **Publish** sub-menu and by clicking **Publish to**. For some locations, you can choose between the following formats:

- Markdown: publish the Markdown text on a website that can interpret it (**GitHub** for instance),
- HTML: publish the file converted to HTML via a Handlebars template (on a blog for example).

## Update a publication

After publishing, StackEdit keeps your file linked to that publication which makes it easy for you to re-publish it. Once you have modified your file and you want to update your publication, click on the **Publish now** button in the navigation bar.

> **Note:** The **Publish now** button is disabled if your file has not been published yet.

## Manage file publication

Since one file can be published to multiple locations, you can list and manage publish locations by clicking **File publication** in the **Publish** sub-menu. This allows you to list and remove publication locations that are linked to your file.


# Markdown extensions

StackEdit extends the standard Markdown syntax by adding extra **Markdown extensions**, providing you with some nice features.

> **ProTip:** You can disable any **Markdown extension** in the **File properties** dialog.


## SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|                |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|


## KaTeX

You can render LaTeX mathematical expressions using [KaTeX](https://khan.github.io/KaTeX/):

The *Gamma function* satisfying $\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$ is via the Euler integral

$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$

> You can find more information about **LaTeX** mathematical expressions [here](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference).


## UML diagrams

You can render UML diagrams using [Mermaid](https://mermaidjs.github.io/). For example, this will produce a sequence diagram:

```mermaid
sequenceDiagram
Alice ->> Bob: Hello Bob, how are you?
Bob-->>John: How about you John?
Bob--x Alice: I am good thanks!
Bob-x John: I am good thanks!
Note right of John: Bob thinks a long<br/>long time, so long<br/>that the text does<br/>not fit on a row.

Bob-->Alice: Checking with John...
Alice->John: Yes... John, how are you?
```

And this will produce a flow chart:

```mermaid
graph LR
A[Square Rect] -- Link text --> B((Circle))
A --> C(Round Rect)
B --> D{Rhombus}
C --> D
```
