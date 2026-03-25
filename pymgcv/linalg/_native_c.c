#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* col_squared_norms(PyObject* self, PyObject* args) {
    PyObject* arr_obj = NULL;
    PyArrayObject* arr = NULL;
    npy_intp rows;
    npy_intp cols;
    npy_intp i;
    npy_intp j;
    npy_intp out_dims[1];
    PyArrayObject* out = NULL;
    double* out_data;

    if (!PyArg_ParseTuple(args, "O", &arr_obj)) {
        return NULL;
    }

    arr = (PyArrayObject*)PyArray_FROM_OTF(arr_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(arr) != 2) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D float64 array.");
        return NULL;
    }

    rows = PyArray_DIM(arr, 0);
    cols = PyArray_DIM(arr, 1);
    out_dims[0] = cols;
    out = (PyArrayObject*)PyArray_ZEROS(1, out_dims, NPY_DOUBLE, 0);
    if (out == NULL) {
        Py_DECREF(arr);
        return NULL;
    }

    out_data = (double*)PyArray_DATA(out);

    for (j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (i = 0; i < rows; ++i) {
            double v = *(double*)PyArray_GETPTR2(arr, i, j);
            sum += v * v;
        }
        out_data[j] = sum;
    }

    Py_DECREF(arr);
    return (PyObject*)out;
}

static PyMethodDef NativeMethods[] = {
    {
        "col_squared_norms",
        col_squared_norms,
        METH_VARARGS,
        "Return squared L2 norm of each column in a 2D float64 array."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_module = {
    PyModuleDef_HEAD_INIT,
    "_native_c",
    "Native C linear algebra kernels for pymgcv.",
    -1,
    NativeMethods
};

PyMODINIT_FUNC PyInit__native_c(void) {
    import_array();
    return PyModule_Create(&native_module);
}
