from quantize.preconditioner import ScalarPreconditioner, DiagonalPreconditioner, BlockwiseHouseholderPreconditioner, ScalarPreconditionerAct


class QuantizationConfig:
    def __init__(self):
        # Quantized Training
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 8
        self.bweight_num_bits = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.biprecision = True

        # Memory Saving
        self.compress_activation = False
        self.activation_compression_bits = 8
        self.initial_bits = 8
        self.alg = 'greedy'

    def activation_preconditioner(self):
        # return lambda x: ForwardPreconditioner(x, self.activation_num_bits)
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)
        # return lambda x: ScalarPreconditioner(x, 16)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)
        # return lambda x: ForwardPreconditioner(x, self.weight_num_bits)
        # return lambda x: DiagonalPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self):
        if self.hadamard:
            return lambda x: BlockwiseHouseholderPreconditioner(x, self.backward_num_bits)
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.backward_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self):
        if self.backward_persample:
            return lambda x: DiagonalPreconditioner(x, self.bweight_num_bits, left=False)
        else:
            return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)


config = QuantizationConfig()
