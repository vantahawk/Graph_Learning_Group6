import numpy as np

def generate_data(as_generator=False, size=10):
    if as_generator:
        # Define a generator function
        def data_generator(size):
            for i in range(size):
                yield i  # Or any other data you want to generate
        return data_generator(size)
    else:
        # Return a normal numpy array
        return np.arange(size)

# Example usage:
# As a generator
gen = generate_data(as_generator=True, size=5)
for val in gen:
    print(val)

# As a normal numpy array
arr = generate_data(as_generator=False, size=5)
print(arr)