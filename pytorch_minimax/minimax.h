enum Reduce { MIN, MAX };
template<Reduce r> torch::Tensor minimax_cuda(torch::Tensor x);
