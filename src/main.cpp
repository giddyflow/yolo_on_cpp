#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// --- НАСТРОЙКИ ДЛЯ YOLO ---
const float CONF_THRESHOLD = 0.7f;    // Порог уверенности для детекции
const float NMS_THRESHOLD = 0.4f;     // Порог для non-maximum suppression
const int INPUT_WIDTH = 416;          // Ширина изображения для входа в сеть
const int INPUT_HEIGHT = 416;         // Высота изображения для входа в сеть

int main() {

    std::string modelWeights = "yolov4-tiny.weights"; 
    std::string modelConfig = "yolov4-tiny.cfg";
    std::string classesFile = "coco.names";

    // 1. Загрузка имен классов
    std::vector<std::string> classes;
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Cannot open classes file: " << classesFile << std::endl;
        return -1;
    }
    std::string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }
    std::cout << "Classes loaded successfully." << std::endl;

    // 2. Загрузка нейронной сети
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    std::cout << "Network loaded successfully." << std::endl;
    
    // 3. Установка бэкенда для вычислений (САМЫЙ ВАЖНЫЙ ШАГ!)
    // Раскомментируйте строки для CUDA, если вы компилировали OpenCV с поддержкой NVIDIA GPU
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    
    // По умолчанию используется CPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Using CPU for inference." << std::endl;


    // --- ИНИЦИАЛИЗАЦИЯ КАМЕРЫ ---
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "CRITICAL ERROR: Cannot open the camera." << std::endl;
        return -1;
    }
    std::cout << "Camera opened successfully. Press 'ESC' to exit." << std::endl;

    cv::Mat frame;

    // --- ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ ---
    while (true) {
        bool isSuccess = cap.read(frame);
        if (!isSuccess || frame.empty()) {
            std::cerr << "ERROR: Failed to capture a frame or stream ended." << std::endl;
            break;
        }

        // --- ПОДГОТОВКА КАДРА И ЗАПУСК YOLO ---
        // Создаем "блоб" из кадра для подачи в сеть
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Получаем предсказания
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // --- ОБРАБОТКА РЕЗУЛЬТАТОВ ---
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (const auto& out : outs) {
            float* data = (float*)out.data;
            for (int i = 0; i < out.rows; ++i, data += out.cols) {
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                if (confidence > CONF_THRESHOLD) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        // Применяем Non-Maximum Suppression для удаления лишних рамок
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        // --- ОТРИСОВКА РЕЗУЛЬТАТОВ НА КАДРЕ ---
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            int classId = classIds[idx];
            
            // Рисуем рамку
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            // Формируем и рисуем подпись
            std::string label = classes[classId] + ": " + cv::format("%.2f", confidences[idx]);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(box.y, labelSize.height);
            cv::rectangle(frame, cv::Point(box.x, top - labelSize.height - 5), 
                          cv::Point(box.x + labelSize.width, top), 
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(box.x, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        
        // Отображаем кадр с результатами
        cv::imshow("YOLO Live Detection", frame);

        if (cv::waitKey(1) == 27) { // Выход по клавише ESC
            std::cout << "ESC key pressed. Exiting..." << std::endl;
            break;
        }
    }

    // Освобождение ресурсов
    cap.release();
    cv::destroyAllWindows();
    return 0;
}