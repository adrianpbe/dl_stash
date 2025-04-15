import tensorflow as tf

import dl_stash.image


def get_rotation_matrix(angles):
    """Creates 2D rotation matrices from angles in radians.
    
    Args:
        angles: Tensor of shape (batch_size,) or (batch_size, 1) containing rotation angles in radians
        
    Returns:
        Tensor of shape (batch_size, 2, 2) containing 2D rotation matrices
    """
    angles = tf.reshape(angles, [-1, 1])
    
    cos = tf.cos(angles)
    sin = tf.sin(angles)
    
    # Stack values into 2x2 rotation matrices
    # [[cos, -sin],
    #  [sin,  cos]]
    matrices = tf.concat([
        tf.concat([cos, -sin], axis=1),
        tf.concat([sin, cos], axis=1)
    ], axis=1)
    
    return tf.reshape(matrices, [-1, 2, 2])


def get_affine_transformation_matrix(rotation, shear, offset, scale, images):
    """Creates affine transformation matrices by composing rotation, shear, offset and scale.
    
    Args:
        rotation: Tensor of shape (batch_size, 1) containing rotation angles in radians
        shear: Tensor of shape (batch_size, 1) containing shear factors 
        offset: Tensor of shape (batch_size, 2) containing x,y translation offsets
        scale: Tensor of shape (batch_size, 2) containing x,y scale factors
        images: Tensor with images (batch_size, height, width, channels)
        
    Returns:
        Tensor of shape (batch_size, 2, 3) containing affine transformation matrices
    """
    R = get_rotation_matrix(rotation)
    
    # Create shear matrices (batch_size, 2, 2)
    # [[1, shear],
    #  [0, 1   ]]
    ones = tf.ones_like(shear)
    zeros = tf.zeros_like(shear)
    S = tf.concat([
        tf.concat([ones, shear], axis=1),
        tf.concat([zeros, ones], axis=1)
    ], axis=1)
    S = tf.reshape(S, [-1, 2, 2])
    
    # Create scale matrices (batch_size, 2, 2)
    # [[scale_x, 0],
    #  [0, scale_y]]
    scale_x = scale[:, 0:1]
    scale_y = scale[:, 1:2]
    scale_matrix = tf.concat([
        tf.concat([scale_x, zeros], axis=1),
        tf.concat([zeros, scale_y], axis=1)
    ], axis=1)
    scale_matrix = tf.reshape(scale_matrix, [-1, 2, 2])
    
    # Combine first rotation, then shear, then scale (batch_size, 2, 2)
    transform = tf.matmul(scale_matrix, tf.matmul(S, R))

    image_shape = tf.shape(images)
    batch_size = image_shape[0]
    height, width = image_shape[1], image_shape[2]
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    center = tf.cast(tf.stack([width / 2, height / 2])[tf.newaxis, ...], rotation.dtype)
    center = tf.tile(center, [batch_size, 1])
    center = tf.expand_dims(center, axis=-1)
    
    # Calculate how much the center moves after transform
    center_displacement = tf.matmul(transform, center) - center
    
    # Compensate for center displacement and add final offset
    final_offset = -center_displacement + tf.expand_dims(offset, axis=-1)

    # Add offset column (batch_size, 2, 3)
    transform_matrix = tf.concat([transform, final_offset], axis=-1)
    
    return transform_matrix


def make_homogeneous_transform(A):
    """Adds a row [0, 0, 1] to each 2x3 affine transformation matrix in the batch.
    
    Args:
        A: Tensor of shape (batch_size, 2, 3) containing affine transformation matrices
        
    Returns:
        Tensor of shape (batch_size, 3, 3) with added [0, 0, 1] row
    """
    batch_size = tf.shape(A)[0]
    last_row = tf.tile(
        tf.constant([[0.0, 0.0, 1.0]], dtype=A.dtype)[tf.newaxis, ...],
        [batch_size, 1, 1]
    )
    return tf.concat([A, last_row], axis=1)


def get_inverse_affine_transformation_matrix(rotation, shear, offset, scale, images):
    """Creates inverse affine transformation matrices by composing rotation, shear, offset and scale in reverse order.
    
    Args:
        rotation: Tensor of shape (batch_size, 1) containing rotation angles in radians
        shear: Tensor of shape (batch_size, 1) containing shear factors 
        offset: Tensor of shape (batch_size, 2) containing x,y translation offsets
        scale: Tensor of shape (batch_size, 2) containing x,y scale factors
        images: Tensor with images (batch_size, height, width, channels)
        
    Returns:
        Tensor of shape (batch_size, 2, 3) containing inverse affine transformation matrices
    """
    A = get_affine_transformation_matrix(-rotation, -shear, tf.zeros_like(offset), 1.0/(scale + 1e-7), images)
    batch_size = tf.shape(rotation)[0]
    identity = tf.tile(
        tf.expand_dims(tf.eye(2), 0), [batch_size, 1, 1]
    )
    offset_matrix = tf.concat([identity, -tf.expand_dims(offset, axis=-1)], -1)
    transform_matrix = tf.matmul(
        make_homogeneous_transform(A), make_homogeneous_transform(offset_matrix))[:, :-1]
    return transform_matrix


def extract_affine_params(image, params):
    """Extracts and normalizes affine transformation parameters from a parameter tensor.
    
    Args:
        image: Tensor of shape (batch_size, height, width, channels)
        params: Tensor of shape (batch_size, 6) containing [rotation, shear, offset_x, offset_y, scale_x, scale_y]
            where rotation is in radians, shear is a factor, offsets are normalized 
            between -1 and 1 relative to image dimensions, and scale factors are positive values.
            
    Returns:
        Tuple of (rotation, shear, normalized_offset, scale) where:
            - rotation: Tensor of shape (batch_size, 1) containing rotation angles in radians
            - shear: Tensor of shape (batch_size, 1) containing shear factors
            - normalized_offset: Tensor of shape (batch_size, 2) containing x,y offsets in pixels
            - scale: Tensor of shape (batch_size, 2) containing x,y scale factors
    """
    height, width = tf.shape(image)[1], tf.shape(image)[2]
    
    # Extract parameters
    rotation = params[:, 0:1]
    shear = params[:, 1:2]
    offset = params[:, 2:4]
    scale = params[:, 4:6]
    
    # Normalize offset by image dimensions
    # offset_x: -1 corresponds to -width, offset_y: 1 corresponds to height
    normalized_offset = tf.stack([
        offset[:, 0] * tf.cast(width, dtype=offset.dtype),
        offset[:, 1] * tf.cast(height, dtype=offset.dtype)
    ], axis=1)
    
    return rotation, shear, normalized_offset, scale


def sample(image, params, sampling_grid):
    """Samples from an image using affine transformation parameters.
    
    Args:
        image: Tensor of shape (batch_size, height, width, channels)
        params: Tensor of shape (batch_size, 6) containing [rotation, shear, offset_x, offset_y, scale_x, scale_y]
        sampling_grid: Tuple of (height, width) for the sampling grid dimensions
        
    Returns:
        Transformed image tensor of shape (batch_size, sampling_grid[0], sampling_grid[1], channels)
    """
    rotation, shear, normalized_offset, scale = extract_affine_params(image, params)

    # Get the transformation matrix 
    transform_matrix = get_affine_transformation_matrix(
        rotation, shear, normalized_offset, scale, image
    )

    transformed_image = dl_stash.image.affine_transform(
        image, 
        transform_matrix, 
        sample_grid=sampling_grid
    )
    
    return transformed_image


def inverse_sampling(image, params, sampling_grid):
    """Samples from an image using the inverse affine transformation given certain parameters.
    
    Args:
        image: Tensor of shape (batch_size, height, width, channels)
        params: Tensor of shape (batch_size, 6) containing [rotation, shear, offset_x, offset_y, scale_x, scale_y]
        sampling_grid: Tuple of (height, width) for the sampling grid dimensions
        
    Returns:
        Transformed image tensor of shape (batch_size, sampling_grid[0], sampling_grid[1], channels)
    """
    rotation, shear, normalized_offset, scale = extract_affine_params(image, params)

    # Get the transformation matrix 
    transform_matrix = get_inverse_affine_transformation_matrix(
        rotation, shear, normalized_offset, scale, image
    )

    transformed_image = dl_stash.image.affine_transform(
        image, 
        transform_matrix, 
        sample_grid=sampling_grid
    )
    
    return transformed_image
