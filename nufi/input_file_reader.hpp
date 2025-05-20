// input_file_reader.hpp
#pragma once

#include <string>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Generic declaration
template<typename T>
T from_string(const std::string& s);

class ConfigReader {
public:
    void load(const std::string& filename);

    template<typename T>
    T read(const std::string& key) const {
        auto it = values.find(key);
        if (it == values.end()) {
            throw std::runtime_error("Key not found: " + key);
        }
        return from_string<T>(it->second);
    }

    template<typename T>
    T read(const std::string& key, const T& default_value) const {
        auto it = values.find(key);
        if (it == values.end()) {
            return default_value;
        }
        return from_string<T>(it->second);
    }

private:
    std::map<std::string, std::string> values;
};

// Specializations
template<>
inline int from_string<int>(const std::string& s) {
    return std::stoi(s);
}

template<>
inline double from_string<double>(const std::string& s) {
    return std::stod(s);
}

template<>
inline std::string from_string<std::string>(const std::string& s) {
    return s;
}

// Implementation of load function
inline void ConfigReader::load(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Unable to open config file: " + filename);
    }

    std::string line;
    while (std::getline(infile, line)) {
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // Trim whitespace
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string val = line.substr(eq_pos + 1);

        // Trim leading/trailing whitespace
        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        val.erase(0, val.find_first_not_of(" \t\r\n"));
        val.erase(val.find_last_not_of(" \t\r\n") + 1);

        if (!key.empty()) {
            values[key] = val;
        }
    }
}
