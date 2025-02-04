#include <iostream>
#include <cmath>
#include <chrono>

template<typename T>
void calculateSineSum(int size) {
    T* array = new T[size];
    T sum = 0;
    
    for (int i = 0; i < size; ++i) {
        array[i] = std::sin(2 * M_PI * i / size);
        sum += array[i];
    }

    std::cout << "Sum: " << sum << std::endl;

    delete[] array;
}

int main() {
    const int size = 10000000; 

#ifdef USE_DOUBLE
    std::cout << "Double ";
    calculateSineSum<double>(size);
#else
    std::cout << "Float ";
    calculateSineSum<float>(size);
#endif

    return 0;
}