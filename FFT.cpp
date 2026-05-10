#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class FFT {
public:
    static bool sizeRight(size_t n) {
        if (n == 0) return false;
        while (n % 2 == 0) n /= 2;
        while (n % 3 == 0) n /= 3;
        while (n % 5 == 0) n /= 5;
        return n == 1;
    }

    // Forward
    static std::vector<std::complex<double>> forward(const std::vector<std::complex<double>>& input) {
        std::vector<std::complex<double>> data = input;
        if (!sizeRight(data.size())) {
            throw std::runtime_error("Unright size");
        }
        transform_dif(data, false);
        return data;
    }

    // Backward
    static std::vector<std::complex<double>> backward(const std::vector<std::complex<double>>& input) {
        std::vector<std::complex<double>> data = input;
        if (!sizeRight(data.size())) {
            throw std::runtime_error("Unright size: length must be composed of factors 2, 3, 5 only");
        }
        transform_dif(data, true);
        size_t N = data.size();
        for (auto& x : data) x /= static_cast<double>(N);
        return data;
    }

private:
    static int smallest_factor(size_t n) {
        if (n % 2 == 0) return 2;
        if (n % 3 == 0) return 3;
        if (n % 5 == 0) return 5;
        return 1;
    }
    static void transform_dif(std::vector<std::complex<double>>& a, bool inverse) {
        const size_t N = a.size();
        if (N <= 1) return;

        const int m = smallest_factor(N);
        const size_t k = N / m;

        std::vector<std::complex<double>> tmp(N);

        for (size_t j = 0; j < m; ++j) {
            for (size_t r = 0; r < k; ++r) {
                std::complex<double> sum = 0.0;
                for (size_t q = 0; q < m; ++q) {
                    size_t idx = q * k + r;
                    double angle = (inverse ? 1.0 : -1.0) * 2.0 * M_PI * j * q / static_cast<double>(m);
                    std::complex<double> w = std::polar(1.0, angle);
                    sum += a[idx] * w;
                }
                tmp[j * k + r] = sum;
            }
        }

        a.swap(tmp);

        for (int j = 0; j < m; ++j) {
            std::vector<std::complex<double>> sub(k);
            for (size_t r = 0; r < k; ++r) {
                sub[r] = a[j * k + r];
            }
            transform_dif(sub, inverse);
            for (size_t r = 0; r < k; ++r) {
                a[j * k + r] = sub[r];
            }
        }
    }
};

int main() {
    try {
        size_t N = 0;
        while (true) {
            std::cout << "Enter the length of the conversion (factors= 2, 3, 5): ";
            std::cin >> N;
            if (FFT::sizeRight(N)) {
                break;
            }
            std::cout << "Error: not all factors " << N << " = 2/3/5)\n";
        }

        std::cout << "FFT size: " << N << "\n";

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<std::complex<double>> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i] = { dist(rng), dist(rng) };
        }

        std::cout << "\nValues:\n";
        for (size_t i = 0; i < N; ++i) {
            std::cout << "Values[" << i << "] = " << input[i] << "\n";
        }

        auto spectrum = FFT::forward(input);
        auto output = FFT::backward(spectrum);

        std::cout << "\nValues after IFT:\n";
        for (size_t i = 0; i < N; ++i) {
            std::cout << "after_IFT[" << i << "] = " << output[i] << "\n";
        }

        double max_error = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = std::abs(input[i] - output[i]);
            if (err > max_error) max_error = err;
        }

        std::cout << "\nMax error: " << std::scientific << max_error << "\n";

    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}