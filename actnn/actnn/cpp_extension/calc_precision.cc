#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

// Greedy algorithm
/*torch::Tensor calc_precision_table(torch::Tensor b, torch::Tensor cost, torch::Tensor C, torch::Tensor w, double target) {
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
        return C_data[l] * (cost_data[l * 8 + b - 1] - cost_data[l * 8 + b - 2]); // negative
    };

    int64_t N = b.size(0);
    double b_sum = 0;
    for (int64_t i = 0; i < N; i++) {
        auto delta = get_obj(i, b_data[i]) / w_data[i];
        q.push(std::make_pair(delta, i));
        b_sum += b_data[i] * w_data[i];
    }

    while (b_sum > target) {        // Pick up the smallest increment (largest decrement)
        assert(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto delta = get_obj(i, b_data[i]) / w_data[i];
            q.push(std::make_pair(delta, i));
        }
    }
    return b;
}*/

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
        for (int b1 = 1; b1 <= b-1; b1++) if (b1 == 1 || b1 == 2 || b1 == 4) {
            float new_cost = C_data[l] * (cost_data[l * 8 + b - 1] - cost_data[l * 8 + b1 - 1]);
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

    auto get_obj = [&](float C, int b) {
        int coeff_1 = ((1 << b) - 1) * ((1 << b) - 1);
        int coeff_2 = ((1 << (b-1)) - 1) * ((1 << (b-1)) - 1);
        return C * (1.0 / coeff_1 - 1.0 / coeff_2);     // negative
    };

    int64_t N = b.size(0);
    double b_sum = 0;
    for (int64_t i = 0; i < N; i++) {
        auto delta = get_obj(C_data[i], b_data[i]) / w_data[i];
        q.push(std::make_pair(delta, i));
        b_sum += b_data[i] * w_data[i];
    }

    while (b_sum > target) {        // Pick up the smallest increment (largest decrement)
        assert(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto delta = get_obj(C_data[i], b_data[i]) / w_data[i];
            q.push(std::make_pair(delta, i));
        }
    }
    return b;
}

struct State {
    float obj;
    int64_t p, b;
};

// Greedy algorithm to find max -a^T theta + beta sqrt(a^T Vinv a)
// where a_i = 2^(-2b_i), s.t., \sum_i b_i w_i <= target
torch::Tensor calc_precision_ucb(torch::Tensor b, torch::Tensor C, double beta, torch::Tensor Vinv,
                                 torch::Tensor w, double target) {
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(C.device().is_cpu(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");
    TORCH_CHECK(w.device().is_cpu(), "w must be a CPU tensor!");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous!");
    TORCH_CHECK(Vinv.device().is_cpu(), "Vinv must be a CPU tensor!");
    TORCH_CHECK(Vinv.is_contiguous(), "Vinv must be contiguous!");

    std::priority_queue<std::pair<float, int64_t>> q;

    auto *b_data = b.data_ptr<int>();
    auto *C_data = C.data_ptr<float>();
    auto *w_data = w.data_ptr<int64_t>();
    auto *Vinv_data = Vinv.data_ptr<float>();

    int64_t N = b.size(0);
    std::vector<float> a(N);
    auto get_a = [&](int b) { return 1.0 / (((1 << b) - 1) * ((1 << b) - 1)); };

    auto compute_obj = [&]() {
        // -lin + beta * sqrt(reg)
        float lin = 0;
        for (int64_t i = 0 ; i < N; i++) {
            a[i] = get_a(b_data[i]);
            lin += a[i] * C_data[i];
        }
        float reg = 0;
        for (int64_t i = 0; i < N; i++)
            for (int64_t j = 0; j < N; j++)
                reg += a[i] * a[j] * Vinv_data[i * N + j];

        return -lin + beta * reg;
    };

    double b_sum = 0;
    auto obj = compute_obj();
//    std::cout << "Initial obj " << obj << std::endl;
    for (int64_t i = 0; i < N; i++) {
        b_data[i]--;
        auto delta = (compute_obj() - obj) / w_data[i]; // amount of reward decrement
        q.push(std::make_pair(delta, i));
        b_data[i]++;
        b_sum += b_data[i] * w_data[i];
//        std::cout << "Candidate " << i << " " << C_data[i] << " " << w_data[i] << " delta " << delta << std::endl;
    }

    while (b_sum > target) {        // Pick up the smallest reward decrement
        assert(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto obj = compute_obj();
//            std::cout << "Picked up " << i << " " << C_data[i] << " " << w_data[i] << " obj " << obj << std::endl;
            b_data[i]--;
            auto delta = (compute_obj() - obj) / w_data[i];
            q.push(std::make_pair(delta, i));
            b_data[i]++;
        }
    }
    obj = compute_obj();
//    std::cout << obj << std::endl;
    return b;
}

// Greedy algorithm to find max -phi(a)^T theta + beta sqrt(phi(a)^T Vinv phi(a))
// where a_i = 2^(-2b_i), s.t., \sum_i b_i w_i <= target
torch::Tensor calc_precision_ucb_g(torch::Tensor b, torch::Tensor C, double beta, torch::Tensor Vinv,
                                 torch::Tensor w, torch::Tensor groups, int G, double target) {
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(C.device().is_cpu(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");
    TORCH_CHECK(w.device().is_cpu(), "w must be a CPU tensor!");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous!");
    TORCH_CHECK(Vinv.device().is_cpu(), "Vinv must be a CPU tensor!");
    TORCH_CHECK(Vinv.is_contiguous(), "Vinv must be contiguous!");
    TORCH_CHECK(groups.device().is_cpu(), "groups must be a CPU tensor!");
    TORCH_CHECK(groups.is_contiguous(), "groups must be contiguous!");

    std::priority_queue<std::pair<float, int64_t>> q;

    auto *b_data = b.data_ptr<int>();
    auto *C_data = C.data_ptr<float>();
    auto *w_data = w.data_ptr<int64_t>();
    auto *groups_data = groups.data_ptr<int64_t>();
    auto *Vinv_data = Vinv.data_ptr<float>();

    int64_t N = b.size(0);
    std::vector<float> a(G);
    auto get_a = [&](int b) { return 1.0 / (((1 << b) - 1) * ((1 << b) - 1)); };

    auto compute_obj = [&]() {
        // -lin + beta * sqrt(reg)
        float lin = 0;
        std::fill(a.begin(), a.end(), 0.0);
        for (int64_t i = 0; i < N; i++)
            a[groups_data[i]] += get_a(b_data[i]);
        for (int64_t i = 0; i < G; i++)
            lin += a[i] * C_data[i];

        float reg = 0;
        for (int64_t i = 0; i < G; i++)
            for (int64_t j = 0; j < G; j++)
                reg += a[i] * a[j] * Vinv_data[i * G + j];

        return -lin + beta * reg;
    };

    double b_sum = 0;
    auto obj = compute_obj();
//    std::cout << "Initial obj " << obj << std::endl;
    for (int64_t i = 0; i < N; i++) {
        b_data[i]--;
        auto delta = (compute_obj() - obj) / w_data[i]; // amount of reward decrement
        q.push(std::make_pair(delta, i));
        b_data[i]++;
        b_sum += b_data[i] * w_data[i];
//        std::cout << "Candidate " << i << " " << C_data[i] << " " << w_data[i] << " delta " << delta << std::endl;
    }

    while (b_sum > target) {        // Pick up the smallest reward decrement
        assert(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto obj = compute_obj();
//            std::cout << "Picked up " << i << " " << C_data[i] << " " << w_data[i] << " obj " << obj << std::endl;
            b_data[i]--;
            auto delta = (compute_obj() - obj) / w_data[i];
            q.push(std::make_pair(delta, i));
            b_data[i]++;
        }
    }
    obj = compute_obj();
//    std::cout << "Obj " << obj << std::endl;
    return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc_precision", &calc_precision, "calc_precision");
  m.def("calc_precision_table", &calc_precision_table, "calc_precision_table");
  m.def("calc_precision_ucb", &calc_precision_ucb, "calc_precision_ucb");
  m.def("calc_precision_ucb_g", &calc_precision_ucb_g, "calc_precision_ucb_g");
}
