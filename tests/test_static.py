class A:
    num_classes = 10

    def __init__(self, a):
        self.a = a


class B(A):
    num_classes = 20

    def __init__(self, a, b):
        super(B, self).__init__(a)
        self.b = b


a = A(1)
b = B(2, 3)

print(B.num_classes)
print(A.num_classes)