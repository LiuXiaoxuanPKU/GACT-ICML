#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

// Greedy algorithm
torch::Tensor calc_precision_table(torch::Tensor b, torch::Tensor cost, torch::Tensor C, torch::Tensor w, double target) {
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(cost.device().is_cpu(), "cost must be a CPU tensor!");
    TORCH_CHECK(cost.is_contiguous(), "cost must be contiguous!");
    TORCH_CHECK(C.device().is_cpu(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");
    TORCH_CHECK(w.device().is_cpu(), "w must be a CPU tensor!");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous!");

    // min \sum_i C_i / (2^b_i - 1)^2, s.t., \sum_i b_i = N b
    std::priority_queue<std::pair<float, int64_t>> q;

    auto *b_data = b.data_ptr<int>();
    auto *cost_data = cost.data_ptr<float>();   // L * 8
    auto *C_data = C.data_ptr<float>();
    auto *w_data = w.data_ptr<int64_t>();

    auto get_obj = [&](int l, int b) {
        int best_b = -1;
        float best_cost = -1e20;
        for (int b1 = 1; b1 <= b-1; b1++) if (b1 == 2 || b1 == 4 || b1 == 8) {
            float new_cost = C_data[l] * (cost_data[l * 32 + b - 1] - cost_data[l * 32 + b1 - 1]);
            if (new_cost > best_cost) {
                best_cost = new_cost;
                best_b = b1;
            }
        }
        return std::make_pair(best_b, best_cost); // negative
    };

    int64_t N = b.size(0);
    double b_sum = 0;
    for (int64_t i = 0; i < N; i++) {
        auto obj = get_obj(i, b_data[i]);
        auto delta = obj.second / w_data[i];
        q.push(std::make_pair(delta, i*100 + obj.first));
        b_sum += b_data[i] * w_data[i];
    }

    while (b_sum > target) {        // Pick up the smallest increment (largest decrement)
        assert(!q.empty());
        auto dat = q.top().second;
        auto i = dat / 100;
        auto nb = dat % 100;
        q.pop();
        b_sum -= (b_data[i] - nb) * w_data[i];
        b_data[i] = nb;
        if (b_data[i] > 1) {
            auto obj = get_obj(i, b_data[i]);
            auto delta = obj.second / w_data[i];
            q.push(std::make_pair(delta, i*100 + obj.first));
        }
    }
    return b;
}

torch::Tensor calc_precision(torch::Tensor b, torch::Tensor C, torch::Tensor w, double target) {
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(C.device().is_cpu(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");
    TORCH_CHECK(w.device().is_cpu(), "w must be a CPU tensor!");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous!");

    // min \sum_i C_i / (2^b_i - 1)^2, s.t., \sum_i b_i = N b
    std::priority_queue<std::pair<float, int64_t>> q;

    auto *b_data = b.data_ptr<int>();
    auto *C_data = C.data_ptr<float>();
    auto *w_data = w.data_ptr<int64_t>();

    std::vector<int> next_b(33);
    next_b[32] = 8;
    next_b[8] = 4;
    next_b[4] = 2;
    next_b[2] = 1;

    auto get_obj = [&](float C, int b) {
        int coeff_1 = ((1 << b) - 1) * ((1 << b) - 1);
        int coeff_2 = ((1 << next_b[b]) - 1) * ((1 << next_b[b]) - 1);
        if (b == 32) return -(double)C / coeff_2;
        else return C * (1.0 / coeff_1 - 1.0 / coeff_2);     // negative
    };

    int64_t N = b.size(0);
    double b_sum = 0;
    for (int64_t i = 0; i < N; i++) if (b_data[i] > 1) {
        auto delta_b = b_data[i] - next_b[b_data[i]];
        auto delta = get_obj(C_data[i], b_data[i]) / (w_data[i] * delta_b);
        q.push(std::make_pair(delta, i));
        b_sum += b_data[i] * w_data[i];
    }

    while (b_sum > target) {        // Pick up the smallest increment (largest decrement)
        assert(!q.empty());
        auto delta = q.top().first;
        auto i = q.top().second;
        q.pop();
        auto delta_b = b_data[i] - next_b[b_data[i]];
        b_data[i] = next_b[b_data[i]];
        b_sum -= w_data[i] * delta_b;
        if (b_data[i] > 1) {
            auto delta_b = b_data[i] - next_b[b_data[i]];
            auto delta = get_obj(C_data[i], b_data[i]) / (w_data[i] * delta_b);
            q.push(std::make_pair(delta, i));
        }
    }
    return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc_precision", &calc_precision, "calc_precision");
  m.def("calc_precision_table", &calc_precision_table, "calc_precision_table");
}