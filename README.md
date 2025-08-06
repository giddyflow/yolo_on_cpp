# YoloLiveDetector: Детектор объектов на C++ и OpenCV

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Проект для распознавания объектов в реальном времени с веб-камеры. В качестве модели используется нейросеть YOLOv4-tiny, работающая на C++ с помощью библиотеки OpenCV. Проект демонстрирует полный цикл: от захвата видео до обработки и визуализации результатов.

## Оглавление
- [Ключевые возможности](#ключевые-возможности)
- [Стек](#Стек)
- [Начало работы](#начало-работы)
  - [Предварительные требования](#предварительные-требования)
  - [Сборка проекта](#сборка-проекта)
- [Использование](#использование)
  - [Конфигурация](#конфигурация)
- [Лицензия](#лицензия)

## Ключевые возможности
- Захват и отображение видеопотока с веб-камеры.
- Высокопроизводительное распознавание объектов с помощью YOLOv4-tiny.
- Отрисовка ограничительных рамок и классов на каждом кадре.
- Поддержка вычислений на CPU и GPU (NVIDIA CUDA).
- Кроссплатформенная сборка с помощью современного CMake.

## Стек
- **Язык:** C++ (Стандарт: C++17)
- **Система сборки:** CMake (версия 3.16 или выше)
- **Сторонние библиотеки:**
  - [OpenCV 4.x](https://opencv.org/): для работы с видео, изображениями и нейронными сетями (модуль `dnn`).

## Начало работы

### Предварительные требования
Для сборки и запуска проекта убедитесь, что у вас установлены:
- [Git](https://git-scm.com/)
- [CMake](https://cmake.org/download/) >= 3.16
- C++ компилятор с поддержкой C++17 (например, GCC/g++ или Clang)
- **OpenCV 4.x**. Рекомендуется сборка из исходников для поддержки GPU.

**Команды для установки зависимостей в Ubuntu:**

1.  **Базовые инструменты:**
    ```bash
    sudo apt-get update
    sudo apt-get install git cmake build-essential
    ```

2.  **Зависимости для сборки OpenCV:**
    ```bash
    sudo apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python3-dev python3-numpy libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev
    ```
    Инструкции по сборке самой библиотеки OpenCV можно найти в ее официальной документации.

### Сборка проекта

1.  **Клонируйте репозиторий:**
    ```bash
    git clone <URL-вашего-репозитория>
    cd <название-папки-проекта>
    ```

2.  **Скачайте файлы модели YOLO:**
    ```bash
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
    ```

3.  **Соберите проект с помощью CMake:**
    ```bash
    cmake -B build
    cmake --build build
    ```
    После успешной сборки исполняемый файл `YoloDemo` и файлы модели появятся в папке `build`.


### Конфигурация
Основная конфигурация (переключение между CPU/GPU) находится в файле `src/main.cpp`.

Найдите этот блок кода для выбора устройства для вычислений:
```cpp
// 3. Установка бэкенда для вычислений
// Раскомментируйте строки для CUDA, если вы компилировали OpenCV с поддержкой NVIDIA GPU
// net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
// net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

// По умолчанию используется CPU
net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
```
После внесения изменений пересоберите проект (```bash cmake --build build```).

Все необходимые файлы (.weights, .cfg, .names) копируются в папку сборки автоматически.

**Для запуска перейдите в папку build и выполните команду:**
```bash
cd build
./YoloDemo
```
    
Нажмите клавишу ESC, чтобы завершить работу программы
