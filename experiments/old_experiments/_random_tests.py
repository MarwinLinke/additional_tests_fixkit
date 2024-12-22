import random

random.seed(-444)

a = [random.randint(0, 100) for i in range(100)]

print(a)
print(list(set(a)))