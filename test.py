import numpy as np

def quaternion_dot(q1, q2):
    # Compute the dot product of two quaternions
    return np.dot(q1, q2)

def check_perpendicularity(qA, qB):
    # Normalize quaternions to ensure correct dot product
    qA = qA / np.linalg.norm(qA)
    qB = qB / np.linalg.norm(qB)
    
    # Compute the dot product
    dot_product = quaternion_dot(qA, qB)
    
    # Absolute value of dot product (to compare deviation from zero)
    deviation = np.abs(dot_product)
    
    return deviation

# Example quaternions A and B
qA = np.array([0.75834996, -0.39924505, -0.43015185,  0.28368665])  # Replace with actual quaternion values
qB = np.array([0.14121688365891152, -0.42428388909962866, 0.7247400071860162, 0.5242069202177444])  # Replace with actual quaternion values

# Check how far B is from being perpendicular to A
deviation = check_perpendicularity(qA, qB)
print(f"Deviation from perpendicularity: {deviation}")

# A value of 0 means the quaternions are perpendicular
