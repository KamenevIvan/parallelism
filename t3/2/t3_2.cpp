#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

template<typename T>
class Server {
public:
    Server() : running(false) {}

    void start() {
        running = true;
        server_thread = std::thread(&Server::process_tasks, this);
    }

    void stop() {
        running = false;
        cv.notify_all();
        if (server_thread.joinable()) {
            server_thread.join();
        }
    }

    size_t add_task(std::function<T()> task) {
        std::unique_lock<std::mutex> lock(mtx);
        size_t id = next_id++;
        task_queue.push({id, std::async(std::launch::deferred, task)});
        cv.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this, id] { return results.find(id) != results.end(); });
        return results[id];
    }

private:
    void process_tasks() {
        while (running) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return !task_queue.empty() || !running; });
            if (!task_queue.empty()) {
                auto task = std::move(task_queue.front()); 
                task_queue.pop();
                lock.unlock();
                T result = task.second.get();
                lock.lock();
                results[task.first] = result;
                cv.notify_all();
            }
        }
    }

    std::thread server_thread;
    bool running;
    std::queue<std::pair<size_t, std::future<T>>> task_queue;
    std::unordered_map<size_t, T> results;
    std::mutex mtx;
    std::condition_variable cv;
    size_t next_id = 0;
};

void client_sin(Server<double>& server, int task_count, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.14, 3.14);

    for (int i = 0; i < task_count; ++i) {
        double arg = dis(gen);
        size_t id = server.add_task([arg] { return std::sin(arg); });
        double result = server.request_result(id);
        file << "sin(" << arg << ") = " << result << std::endl;
    }
}

void client_sqrt(Server<double>& server, int task_count, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);

    for (int i = 0; i < task_count; ++i) {
        double arg = dis(gen);
        size_t id = server.add_task([arg] { return std::sqrt(arg); });
        double result = server.request_result(id);
        file << "sqrt(" << arg << ") = " << result << std::endl;
    }
}

void client_pow(Server<double>& server, int task_count, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1, 10);

    for (int i = 0; i < task_count; ++i) {
        double base = dis(gen);
        double exp = dis(gen);
        size_t id = server.add_task([base, exp] { return std::pow(base, exp); });
        double result = server.request_result(id);
        file << base << "^" << exp << " = " << result << std::endl;
    }
}

void test_results(const std::string& filename, const std::string& task_type) {
    std::ifstream file(filename);
    std::string line;
    int errCounter = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double arg1, arg2, result;
        char eq;

        if (task_type == "sin") {
            std::string sin_label;
            iss >> sin_label >> arg1 >> eq >> result;
            double expected = std::sin(arg1); 
            if (std::abs(result - expected) > 1e-8) {
                std::cerr << "Error in " << filename << ": " << line << " expected " << expected << std::endl;
            }
        } else if (task_type == "sqrt") {
            std::string sqrt_label;
            iss >> sqrt_label >> arg1 >> eq >> result;
            double expected = std::sqrt(arg1); 
            if (std::abs(result - expected) > 1e-8) {
                std::cerr << "Error in " << filename << ": " << line << " expected " << expected << std::endl;
            }
        } else if (task_type == "pow") {
            
            char caret;
            iss >> arg1 >> caret >> arg2 >> eq >> result;
            double expected = std::pow(arg1, arg2);
            if (std::abs(result - expected) > 1e-1) {
                std::cerr << "Error " << errCounter <<" in " << filename << ": " << line << " expected " << expected << std::endl;
                errCounter+=1;
            }
        } else {
            std::cerr << "Unknown task type: " << task_type << std::endl;
            return;
        }
    }
}

int main() {
    Server<double> server;
    server.start();

    std::thread client1(client_sin, std::ref(server), 1000, "sin_results.txt");
    std::thread client2(client_sqrt, std::ref(server), 1000, "sqrt_results.txt");
    std::thread client3(client_pow, std::ref(server), 1000, "pow_results.txt");

    client1.join();
    client2.join();
    client3.join();

    server.stop();

    test_results("sin_results.txt", "sin");
    test_results("sqrt_results.txt", "sqrt");
    test_results("pow_results.txt", "pow");

    std::cout << "Testing completed." << std::endl;
    return 0;
}