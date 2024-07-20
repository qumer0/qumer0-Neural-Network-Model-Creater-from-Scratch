import numpy as np

def MaxPooling2D(input_matrix, size=2, stride=2):
    output_shape = ((input_matrix.shape[0] - size) // stride + 1, (input_matrix.shape[1] - size) // stride + 1)
    pooled_matrix = np.zeros(output_shape)
    max_indices = np.zeros(output_shape, dtype=tuple)

    for i in range(0, input_matrix.shape[0] - size + 1, stride):
        for j in range(0, input_matrix.shape[1] - size + 1, stride):
            patch = input_matrix[i:i+size, j:j+size]
            max_indices[i // stride, j // stride] = np.unravel_index(np.argmax(patch, axis=None), patch.shape)
            pooled_matrix[i // stride, j // stride] = np.max(patch)
    
    return pooled_matrix, max_indices

def MaxUnPool2D(pooled_matrix, max_indices, original_shape, size=2, stride=2):
    unpooled_matrix = np.zeros(original_shape)
    
    for i in range(pooled_matrix.shape[0]):
        for j in range(pooled_matrix.shape[1]):
            unpooled_matrix[i*stride + max_indices[i, j][0], j*stride + max_indices[i, j][1]] = pooled_matrix[i, j]
    
    return unpooled_matrix


def AvaragePooling2D(input_matrix, size=2, stride=2):
    output_shape = ((input_matrix.shape[0] - size) // stride + 1, (input_matrix.shape[1] - size) // stride + 1)
    pooled_matrix = np.zeros(output_shape)

    for i in range(0, input_matrix.shape[0] - size + 1, stride):
        for j in range(0, input_matrix.shape[1] - size + 1, stride):
            pooled_matrix[i // stride, j // stride] = np.mean(input_matrix[i:i+size, j:j+size])
    
    return pooled_matrix

def AvarageUnPool2D(pooled_matrix, original_shape, size=2, stride=2):
    unpooled_matrix = np.zeros(original_shape)
    count_matrix = np.zeros(original_shape)
    
    for i in range(pooled_matrix.shape[0]):
        for j in range(pooled_matrix.shape[1]):
            unpooled_matrix[i*stride:i*stride+size, j*stride:j*stride+size] += pooled_matrix[i, j]
            count_matrix[i*stride:i*stride+size, j*stride:j*stride+size] += 1
    
    return unpooled_matrix / count_matrix


def MinPooling2D(input_matrix, size=2, stride=2):
    output_shape = ((input_matrix.shape[0] - size) // stride + 1, (input_matrix.shape[1] - size) // stride + 1)
    pooled_matrix = np.zeros(output_shape)
    min_indices = np.zeros(output_shape, dtype=tuple)

    for i in range(0, input_matrix.shape[0] - size + 1, stride):
        for j in range(0, input_matrix.shape[1] - size + 1, stride):
            patch = input_matrix[i:i+size, j:j+size]
            min_indices[i // stride, j // stride] = np.unravel_index(np.argmin(patch, axis=None), patch.shape)
            pooled_matrix[i // stride, j // stride] = np.min(patch)
    
    return pooled_matrix, min_indices

def MinUnPool2D(pooled_matrix, min_indices, original_shape, size=2, stride=2):
    unpooled_matrix = np.zeros(original_shape)
    
    for i in range(pooled_matrix.shape[0]):
        for j in range(pooled_matrix.shape[1]):
            unpooled_matrix[i*stride + min_indices[i, j][0], j*stride + min_indices[i, j][1]] = pooled_matrix[i, j]
    
    return unpooled_matrix