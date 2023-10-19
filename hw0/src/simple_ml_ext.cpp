// #include <cstddef>
// #include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;
/**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
void matMul(const float *x, float *theta, float *o, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            o[i * k + j] = 0;
            for (int t = 0; t < n; t++) {
                // x: i t   theta: t j   o:i j
                // x: m n   theta: n k   o:m k
                o[i * k + j] += x[i * n + t] * theta[t * k + j];
            }
        }
    }
}
void sigmoid(float *mat, int m, int k) {
    for (int i = 0; i < m; i++) {
        float temp = 0;
        for (int j = 0; j < k; j++) {
            mat[i * k + j] = std::exp(mat[i * k + j]);
            temp += mat[i * k + j];
        }
        for (int j = 0; j < k; j++) {
            mat[i * k + j] /= temp;
        }
    }
}
void hotDeal(float *mat, int m, int k, const unsigned char *y) {
    for (int i = 0; i < m; i++) {
        char label = y[i];
        mat[i * k + label] -= 1;
    }
}
// x:m*n  o:m*k
// x的转置n*m o:m*k
// dg n*k
void calDg(const float *x, float *o, float *dg, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            int index = i * k + j;
            dg[index] = 0;
            for (int t = 0; t < m; t++) {
                dg[index] += x[t * n + i] * o[t * k + j];
            }
        }
    }
}
void updateTheta(float *theta, float *dg, int n, int k, float lr, float batch) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            theta[i * k + j] -= lr / batch * dg[i * k + j];
        }
    }
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    int batch_num = m / batch;
    float *o = (float *)malloc(sizeof(float) * m * k);
    float *dg = (float *)malloc(sizeof(float) * n * k);
    for (int i = 0; i < batch_num; i++) {
        const float *x_d = X + n * batch * i;
        const unsigned char *y_d = y + batch * i;
        matMul(x_d, theta, o, batch, n, k);
        sigmoid(o, batch, k);
        hotDeal(o, batch, k, y_d);
        // theta-=lr/batch*(np.dot(x_d.T,z-oh))
        // theta-=lr/batch*(x_d.T,o)
        calDg(x_d, o, dg, batch, n, k);
        updateTheta(theta, dg, n, k, lr, batch);
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
