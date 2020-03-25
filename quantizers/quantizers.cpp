#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_transform", &get_transform, "get_transform");
}
