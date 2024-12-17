#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

template <typename T>
std::vector<std::vector<T>> readCsvToVectorOfVectors(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::vector<std::vector<T>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<T> row;
        std::istringstream stream(line);
        std::string value;

        while (std::getline(stream, value, ',')) {
            row.push_back(static_cast<T>(std::stod(value)));
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

template <typename T>
std::vector<T> readCsvToVector(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::vector<T> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string value;  

        while (std::getline(stream, value, ',')) {
            data.push_back(static_cast<T>(std::stod(value)));
        }
    }

    file.close();
    return data;
}