#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

struct KeyValue {
    int k;
    float v;
};

std::vector<torch::Tensor> get_transform(torch::Tensor mvec,
                            std::vector<torch::Tensor> Q,
                            std::vector<float> Qmax) {
    TORCH_CHECK(!mvec.type().is_cuda(), "mvec must be a CPU tensor!");
    TORCH_CHECK(mvec.is_contiguous(), "mvec must be contiguous!")

    auto N = mvec.size(0);
    auto *mvec_data = mvec.data_ptr<float>();

    std::vector<KeyValue> rank(N);
    float total_values = 0;
    for (int i = 0; i < N; i++) {
        rank[i].v = mvec_data[i];
        rank[i].k = i;
        total_values += mvec_data[i];
    }
    std::sort(rank.begin(), rank.end(),
              [](KeyValue &a, KeyValue &b) { return a.v > b.v; });

    int num_zeros = 0;
    while (1) {
        num_zeros++;
        total_values -= rank[N - num_zeros].v;
        float num = num_zeros * rank[N - num_zeros - 1].v / total_values;
        if (num > 1) break;
    }

    int num_nonzeros = N - num_zeros;
    std::vector<int> nums(num_nonzeros);
    float sum = 0;
    for (int i = 0; i < num_nonzeros; i++) {
        sum += num_zeros * rank[i].v / total_values;
        nums[i] = roundf(sum);
    }
    TORCH_CHECK(nums[num_nonzeros - 1] == num_zeros, "Number mismatch");

    auto T = torch::zeros({N, N}, mvec.options());
    std::vector<float> all_s(N), all_s2(N), all_s2_inv(N);

    auto *T_data = T.data_ptr<float>();
    std::vector<int> indices;
    int index_cnt = 0;
    int cnt = num_nonzeros;

    for (int i = 0; i < num_nonzeros; i++) {
        indices.push_back(rank[i].k);
        float lambda_1 = rank[i].v;
        float lambda_2 = rank[cnt].v;
        int sz = num_nonzeros + nums[i] - cnt + 1;
        TORCH_CHECK(sz > 0, "sz is zero");
        auto q = Q[sz];
        auto qmax = Qmax[sz];
        float w1 = lambda_1 / sqrt(float(sz)), w2 = lambda_2 * qmax;
        float s1 = pow(w1, -1.0 / 3), s2 = pow(w2 / (sz - 1), -1.0 / 3);
        float s_norm = sqrt((1 / s1 / s1) + (1 / s2 / s2));
        s1 *= s_norm;
        s2 *= s_norm;

        all_s[index_cnt] = s1;
        for (int j = index_cnt + 1; j < index_cnt + sz; j++)
            all_s[j] = s2;

        auto *q_data = q.data_ptr<float>();
        for (int r = 0; r < sz; r++)
            for (int c = 0; c < sz; c++)
                T_data[(index_cnt + r) * N + index_cnt + c] = q_data[r * sz + c];
//        for (int r = 0; r < sz; r++)
//            T_data[(index_cnt + r) * N + index_cnt + r] = 1.0;

//        std::cout << index_cnt << ' ' << index_cnt + sz << std::endl;

        index_cnt += sz;
        for (int j = cnt; j < num_nonzeros + nums[i]; j++)
            indices.push_back(rank[j].k);
        cnt = num_nonzeros + nums[i];
    }

    TORCH_CHECK(indices.size() == N, "Indices is not N")

    auto T2 = torch::zeros({N, N}, mvec.options());
    auto *T2_data = T2.data_ptr<float>();
    auto T2_inv = torch::zeros({N, N}, mvec.options());
    auto *T2_inv_data = T2_inv.data_ptr<float>();

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            T2_data[indices[r] * N + indices[c]] = T_data[r * N + c];
            T2_inv_data[indices[c] * N + indices[r]] = T_data[r * N + c];
        }
        all_s2[indices[r]] = all_s[r];
        all_s2_inv[indices[r]] = 1.0 / all_s[r];
    }

    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            T2_data[r * N + c] *= all_s2[c];
            T2_inv_data[r * N + c] *= all_s2_inv[r];
        }

    return {T2, T2_inv};
}

// Greedy algorithm
torch::Tensor calc_precision(torch::Tensor b, torch::Tensor C, int target) {
    TORCH_CHECK(!b.type().is_cuda(), "b must be a CPU tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(!C.type().is_cuda(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");

    // min \sum_i C_i / (2^b_i - 1)^2, s.t., \sum_i b_i = N b
    std::priority_queue<std::pair<float, int>> q;

    auto *b_data = b.data_ptr<int>();
    auto *C_data = C.data_ptr<float>();

    auto get_obj = [&](float C, int b) {
        int coeff_1 = ((1 << b) - 1) * ((1 << b) - 1);
        int coeff_2 = ((1 << (b-1)) - 1) * ((1 << (b-1)) - 1);
        return C * (1.0 / coeff_1 - 1.0 / coeff_2);     // negative
    };

    int N = b.size(0);
    int b_sum = 0;
    for (int i = 0; i < N; i++) {
        auto delta = get_obj(C_data[i], b_data[i]);
        q.push(std::make_pair(delta, i));
        b_sum += b_data[i];
    }

    while (b_sum > target) {
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum--;
        if (b_data[i] > 1) {
            auto delta = get_obj(C_data[i], b_data[i]);
            q.push(std::make_pair(delta, i));
        }
    }
    return b;
}

struct State {
    float obj;
    int p, b;
};

// Dynamic programming
std::pair<torch::Tensor, torch::Tensor> calc_precision_dp(torch::Tensor A, torch::Tensor C, int max_b, int target, int states) {
    using namespace std;

    TORCH_CHECK(!A.type().is_cuda(), "A must be a CPU tensor!");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous!");
    TORCH_CHECK(!C.type().is_cuda(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");
    // min \sum_i (1-p_i)/p_i A_i + C_i / (p_i B_i^2),
    // s.t. \sum_i p_i b_i = N b
    // where B_i = 2^b_i - 1, and p_i takes ``states'' discrete states.

    // We solve with dynamic programming, where
    // f[i, b] is the minimum objective function using the first i terms and b/s bits,
    // where i \in [0, N] and b \in [0, N * b * states]
    // the time complexity is O(N^2bs) states and O(bs) transitions
    // O((Nbs)^2) in total, where N=128, b=2, and s=2, (Nbs)^2 = 262144

    int N  = A.size(0);
    auto *A_data = A.data_ptr<float>();
    auto *C_data = C.data_ptr<float>();
    int total_states = target * N * states;

    // Initialize
    std::vector<std::vector<State>> f(N+1);
    for (auto &v: f) {
        v.resize(total_states + 1);
        for (auto &state: v)
            state.obj = 1e20;
    }
    f[0][0].obj = 0;
//    cout << "Initialized " << total_states << endl;

    for (int i = 1; i <= N; i++) {
        // Moving from f[i-1] to f[i]
        for (int b = 0; b < total_states; b++) {
            auto &old_state = f[i-1][b];

            for (int b0 = 1; b0 <= max_b; b0++)
                for (int p = 1; p <= states; p++)
                    if (b + b0 * p <= total_states) {
                        auto &new_state = f[i][b + b0 * p];
                        float p0 = (float)p / states;
                        float B = (1<<b0) - 1;
                        auto delta = (1 - p0) / p0 * A_data[i-1] + C_data[i-1] / (p0 * B * B);
                        if (old_state.obj + delta < new_state.obj) {
                            new_state.obj = old_state.obj + delta;
                            new_state.p = p;
                            new_state.b = b0;
                        }
                    }
        }
    }
//    cout << "DP Finished " << f[N][total_states].obj << endl;

    // Backtrace
    auto b_vec = torch::zeros({N}, A.options());
    auto p_vec = torch::zeros({N}, A.options());
    auto *b_data = b_vec.data_ptr<float>();
    auto *p_data = p_vec.data_ptr<float>();
    int current_state = total_states;
    for (int i = N; i > 0; i--) {
        auto &state = f[i][current_state];
        b_vec[i-1] = state.b;
        p_vec[i-1] = (float)state.p / states;
        current_state -= state.b * state.p;
    }
    TORCH_CHECK(current_state==0, "DP Failed: no path to initial state!");

    return std::make_pair(b_vec, p_vec);
}

//// Newton's method
//std::pair<torch::Tensor, torch::Tensor> calc_precision_newton(torch::Tensor A, torch::Tensor C, int max_b, int target, int states) {
//    // min \sum_i (1-p_i)/p_i A_i + C_i / (p_i B_i^2),
//    // s.t. \sum_i p_i b_i = N b
//    // where B_i = 2^b_i - 1, and p_i takes ``states'' discrete states.
//
//
//}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_transform", &get_transform, "get_transform");
  m.def("calc_precision", &calc_precision, "calc_precision");
  m.def("calc_precision_dp", &calc_precision_dp, "calc_precision_dp");
}
