# estimate_transform_tf

import tensorflow as tf
import numpy as np


def compute_similarity_transform_tf(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0_norm = points_static - tf.reduce_mean(points_static,axis=-2,keepdims=True)
    p1_norm = points_to_transform - tf.reduce_mean(points_to_transform,axis=-2,keepdims=True)
    covariance_matrix = tf.matmul(p0_norm,p1_norm,transpose_a=True)
    s, u, v = tf.linalg.svd(covariance_matrix,full_matrices=True)
    R = tf.matmul(u,v,transpose_b=True)
    return R,covariance_matrix

def compute_similarity_transform(points_static, points_to_transform):
    #http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3,1)
    t1 = -np.mean(p1, axis=1).reshape(-1,3,1)
    t_final = t1 -t0

    p0c = p0+t0
    p1c = p1+t1

    covariance_matrix = p0c.dot(p1c.T) # 3*68 68*3
    U,S,V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:,2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0)**2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0)**2))

    s = (rms_d0/rms_d1)
    P = np.c_[s*np.eye(3).dot(R), t_final]
    return R,covariance_matrix
if __name__ == '__main__':

    p0 = np.random.random((68,3))
    p1 = np.random.random((2,68,3))
    R,covariance_matrix = compute_similarity_transform(p0,p1)

    p0_tf = tf.Variable(p0)
    p1_tf = tf.Variable(p1)
    R_tf,covariance_matrix_tf = compute_similarity_transform_tf(p0_tf,p1_tf)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        R_tf_out,covariance_matrix_tf = sess.run([R_tf,covariance_matrix_tf])
    print(R)
    # print(covariance_matrix)
    print(R_tf_out)
    # print(covariance_matrix_tf)
