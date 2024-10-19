#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>
#define WITHOUT_NUMPY
#include "matplotlibcpp.h"
#define COORDINATES_DISTRIBUTION 1000.f
#define WEIGHT_DISTRIBUTION 2.f
#define EPOCHES_NUM 200
#define lambda 0.0001

namespace plt = matplotlibcpp;

struct Point {
    double x;
    double y;
    inline Point(double x, double y):x(x), y(y) {
    }
};

struct data {
    Point point;
    double expected;  
};

double random(double min, double max, std::mt19937& rnd) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rnd);
}

Point generate_point(std::mt19937& rnd) {
    return Point(random(COORDINATES_DISTRIBUTION, -COORDINATES_DISTRIBUTION, rnd), random(COORDINATES_DISTRIBUTION, -COORDINATES_DISTRIBUTION, rnd));
}

double generate_answer(Point p) {
    return (p.y > -p.x) ? 1.f: 0.f;
}

std::vector<data> generate_dataset(size_t dataset_size, std::mt19937& rnd) {
    std::vector<data> dataset;
    for (size_t i = 0; i < dataset_size; ++i) {
        Point point = generate_point(rnd);
        double answer = generate_answer(point);
        dataset.push_back({point, answer});
    }
    return dataset;
}

double accuracy(std::vector<double> predicts,  std::vector<double> expected) {
    double correct = 0;
    for (size_t i = 0; i < predicts.size(); ++i) {
        double rounded_pred = (predicts[i] > 0.5) ? 1: 0;
        if (expected[i]  - rounded_pred == 0) {
            correct += 1;
        } 
    }
    return correct/predicts.size();
}

std::pair<std::vector<Point>, std::vector<double>> split_dataset(std::vector<data> dataset) {
    std::pair<std::vector<Point>, std::vector<double>> data_expected;
    std::vector<Point> points;
    std::vector<double> expected;
    for (auto& data: dataset) {
        points.push_back(data.point);
        expected.push_back(data.expected);
    }
    data_expected.first = points;
    data_expected.second = expected;
    return data_expected;
}

class NeuralNetwork {
public:
    NeuralNetwork(std::mt19937& rnd): random_generator(rnd) {
        std::cout << "You have created a NeuralNetwork" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            weights.emplace_back(random(-WEIGHT_DISTRIBUTION, WEIGHT_DISTRIBUTION, rnd));
        }
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double sigmoid_derrivative(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    void print_weights() {
        std::for_each(weights.begin(), weights.end(), [](double x){std::cout << x << std::endl;});
    }

    double predict(Point point) {
        return sigmoid(f(point.x, point.y));
    }

    std::vector<double> predict(std::vector<Point> points) {
        std::vector<double> result;
        for (size_t i = 0; i < points.size(); ++i) {
            result.push_back(predict(points[i]));
        }
        return result;
    }

    std::pair<std::vector<double>, double> gradient_calculate(Point point, double expected) {
        std::vector<double> grad;
        double error = sigmoid(f(point.x, point.y)) - expected;
        double derrivative = sigmoid_derrivative(f(point.x, point.y));
        grad.emplace_back(2*error*derrivative*weights[2]*point.x);
        grad.emplace_back(2*error*derrivative*weights[3]*point.y);
        grad.emplace_back(2*error*derrivative*weights[0]*point.x);
        grad.emplace_back(2*error*derrivative*weights[1]*point.y);
        return std::make_pair(grad, error*error);
    }

    double f(double x1, double x2) {
        return weights[0]*weights[2]*x1 + weights[3]*weights[1]*x2;
    }

    void back_propagation(std::vector<data> dataset) {
        for (size_t epoches = 0; epoches < EPOCHES_NUM; ++epoches) {
            std::shuffle(dataset.begin(), dataset.end(), random_generator);
            for (size_t j = 0; j < dataset.size(); ++j) {
                auto grad = gradient_calculate(dataset[j].point, dataset[j].expected);
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] -= lambda*grad.first[i];
                }
            }
            auto data_expected = split_dataset(dataset);
            acc.first.push_back(accuracy(predict(data_expected.first), data_expected.second));
            acc.second.push_back(epoches);
            std::cout << acc.first.back() << std::endl;
            std::ostringstream oss;
            oss << "./plots/plot" << std::setw(3) << std::setfill('0') << epoches << ".png";
            std::string filename = oss.str();
            plt::clf();
            plt::plot(acc.second, acc.first);
            plt::xlabel("epoch num");
            plt::ylabel("accuracy");
            plt::title("traning visualization");
            plt::save(filename);
        }
    }

    std::pair<std::vector<double>, std::vector<double>>& get_accuracy () {
        return acc;
    }

    std::vector<double>& get_squared_errors() {
        return squared_errors;
    }
private: 
    std::vector<double> weights;
    std::pair<std::vector<double>, std::vector<double>> acc;
    std::vector<double> squared_errors;
    std::mt19937& random_generator;
};

int main() {
    std::mt19937 rnd{std::random_device{}()};
    NeuralNetwork ai(rnd);

    std::vector<data> dataset = generate_dataset(10000, rnd);
    ai.back_propagation(dataset);
    
    std::vector<data> validation = generate_dataset(100, rnd);
    auto data_expected = split_dataset(validation);
    std::vector<double> predicts = ai.predict(data_expected.first);

    std::cout << accuracy(predicts, data_expected.second) << std::endl;
    
    std::vector<double> below_line_x, below_line_y;
    std::vector<double> above_line_x, above_line_y;
    for (size_t i = 0; i < validation.size(); ++i) {
        double rounded_pred = (predicts[i] > 0.5) ? 1: 0;
        if (rounded_pred == 1) {
            above_line_x.push_back(validation[i].point.x);
            above_line_y.push_back(validation[i].point.y);
        }
        else {
            below_line_x.push_back(validation[i].point.x);
            below_line_y.push_back(validation[i].point.y);
        }
    }
    plt::figure_size(640, 480);
    plt::scatter(above_line_x, above_line_y, 50.0, {{"color", "blue"}, {"label", "Above y = -x"}});
    plt::scatter(below_line_x, below_line_y, 50.0, {{"color", "red"}, {"label", "Below y = -x"}});
    std::vector<double> line_x = {-1000, 1000};
    std::vector<double> line_y = {1000, -1000};
    plt::plot(line_x, line_y, "k");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Visualization of Points Above and Below y = -x");
    plt::legend();
    system("ffmpeg -y -f image2 -i ./plots/plot%3d.png ./plots/plot.mp4 2> /dev/null");
    system("ffmpeg -y -f image2 -i ./plots/plot%3d.png ./plots/plot.gif 2> /dev/null");
    system("rm -rf ./plots/*.png");
    plt::save("./plots/validation_plot.png");
    return 0;
}