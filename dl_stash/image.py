import tensorflow as tf


def bilinear_interpolate(image, x_coords, y_coords):
    """Performs bilinear interpolation on an image given sampling coordinates.
    
    Args:
        image: Tensor of shape (batch_size, height, width, channels)
        x_coords: Tensor of shape (batch_size, height, width) containing x coordinates to sample
        y_coords: Tensor of shape (batch_size, height, width) containing y coordinates to sample

    Returns:
        Interpolated values tensor of shape (batch_size, height, width, channels)
    """
    shape = tf.shape(image)
    batch_size, height, width = shape[0], shape[1], shape[2]
    
    # Clip coordinates to valid image bounds
    x = tf.clip_by_value(x_coords, 0, tf.cast(width - 1, x_coords.dtype))
    y = tf.clip_by_value(y_coords, 0, tf.cast(height - 1, y_coords.dtype))
    
    # Get corner coordinates
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1
    
    # Clip again to ensure valid indices
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)
    
    # Get weights for bilinear interpolation
    x0_float = tf.cast(x0, x_coords.dtype)
    x1_float = tf.cast(x1, x_coords.dtype)
    y0_float = tf.cast(y0, y_coords.dtype)
    y1_float = tf.cast(y1, y_coords.dtype)
    
    wa = (x1_float - x) * (y1_float - y)
    wb = (x1_float - x) * (y - y0_float)
    wc = (x - x0_float) * (y1_float - y)
    wd = (x - x0_float) * (y - y0_float)
    
    # Gather corner values
    batch_idx = tf.range(batch_size)
    batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
    batch_idx = tf.broadcast_to(batch_idx, tf.shape(x0))
    indices_a = tf.stack([batch_idx, y0, x0], axis=-1)
    indices_b = tf.stack([batch_idx, y1, x0], axis=-1)
    indices_c = tf.stack([batch_idx, y0, x1], axis=-1)
    indices_d = tf.stack([batch_idx, y1, x1], axis=-1)

    pixel_values_a = tf.gather_nd(image, indices_a)
    pixel_values_b = tf.gather_nd(image, indices_b)
    pixel_values_c = tf.gather_nd(image, indices_c)
    pixel_values_d = tf.gather_nd(image, indices_d)
    
    # Apply weights
    wa = tf.expand_dims(wa, -1)
    wb = tf.expand_dims(wb, -1)
    wc = tf.expand_dims(wc, -1)
    wd = tf.expand_dims(wd, -1)
    
    interpolated = tf.add_n([
        wa * pixel_values_a,
        wb * pixel_values_b,
        wc * pixel_values_c,
        wd * pixel_values_d
    ])
    
    return interpolated


def affine_transform(image, transform_matrix):
    """Applies an affine transformation to images using bilinear interpolation.
    
    Args:
        image: Tensor of shape (batch_size, height, width, channels)
        transform_matrix: Tensor of shape (batch_size, 2, 3) containing affine transform matrices
        
    Returns:
        Transformed image tensor of same shape as input
    """
    shape = tf.shape(image)
    batch_size, height, width = shape[0], shape[1], shape[2]
    
    x = tf.range(width, dtype=tf.float32)
    y = tf.range(height, dtype=tf.float32)
    x_t, y_t = tf.meshgrid(x, y)

    # Creates a flat (batch_size, 3) tensor of coordinates to be transformed
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, [batch_size, 1, 1])
    
    batch_grids = tf.matmul(transform_matrix, sampling_grid)
    
    # Reshape x and y coordinates to (batch_size, height, width), required by interpolation
    x_s = tf.reshape(batch_grids[:, 0, :], [batch_size, height, width])
    y_s = tf.reshape(batch_grids[:, 1, :], [batch_size, height, width])
    
    transformed_image = bilinear_interpolate(image, x_s, y_s)
    return transformed_image
