#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "math.h"
#include "digamma.c"
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"


static PyObject* lda_update(PyObject *self, PyObject *args);


static PyMethodDef LDAMethods[] = {
    {"lda_update", lda_update, METH_VARARGS, "LDA document update"},
    {NULL, NULL, 0, NULL}
};



static PyObject* lda_update(PyObject *self, PyObject *args)
{
	long double ld1, ld2;

    npy_float64 **gammas_data, **rgammas_data, **dgammas_data;
	npy_float64 **alphas_data, **dwp_data, **dwc_data, *ww_data, *dg_data;
	npy_float64 *alpha_loc, *gamma_loc, *rgamma_loc, *dgamma_loc, *dwp_loc, *dwc_loc;
	npy_bool *isdone_data;

    PyArrayObject *didx, *gammas, *rgammas, *dgammas, *alphas, *dwp, *dwc, *ww, *isdone, *idxrem, *dg;

	npy_int num_documents, num_topics, num_entries;
	npy_intp *gammas_dims, *dwp_dims, *didx_dims;
	npy_intp *didx_data, *idxrem_data, didx_loc;
	npy_int i, j, k;
	double delta, ww_loc;
	int not_done, tot_its, max_its, wcnt, ecnt, loc_change;

    if(!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!id",
    								 &PyArray_Type, &didx,
    								 &PyArray_Type, &gammas,
    								 &PyArray_Type, &rgammas,
    								 &PyArray_Type, &dgammas,
    								 &PyArray_Type, &alphas,
    								 &PyArray_Type, &dwp,
    								 &PyArray_Type, &dwc,
    								 &PyArray_Type, &ww,
    								 &PyArray_Type, &isdone,
    								 &PyArray_Type, &idxrem,
    								 &PyArray_Type, &dg,
    								 &max_its,
    								 &delta
    								 )) {
        return NULL;
    }


	Py_INCREF(didx);
	Py_INCREF(gammas);
	Py_INCREF(rgammas);
	Py_INCREF(dgammas);
	Py_INCREF(alphas);
	Py_INCREF(dwp);
	Py_INCREF(dwc);
	Py_INCREF(ww);
	Py_INCREF(dg);
	Py_INCREF(isdone);
	Py_INCREF(idxrem);

	gammas_dims = PyArray_DIMS(gammas);
	dwp_dims    = PyArray_DIMS(dwp);
	didx_dims   = PyArray_DIMS(didx);


	PyArray_Descr *descr = PyArray_DescrFromType(NPY_FLOAT64);


	PyArray_AsCArray((PyObject **)&alphas,  (void**) &alphas_data,  gammas_dims, 2, descr);
	PyArray_AsCArray((PyObject **)&gammas, (void**) &gammas_data,  gammas_dims, 2, descr);
	PyArray_AsCArray((PyObject **)&rgammas, (void**)&rgammas_data, gammas_dims, 2, descr);
	PyArray_AsCArray((PyObject **)&dgammas, (void**)&dgammas_data, gammas_dims, 2, descr);
	PyArray_AsCArray((PyObject **)&dwp, (void**)&dwp_data, dwp_dims, 2, descr);
	PyArray_AsCArray((PyObject **)&dwc, (void**)&dwc_data, dwp_dims, 2, descr);


	isdone_data = (npy_bool *)PyArray_DATA(isdone);
	didx_data = (npy_intp *)PyArray_DATA(didx);
	idxrem_data = (npy_intp *)PyArray_DATA(idxrem);
	ww_data = (npy_float64 *)PyArray_DATA(ww);
	dg_data = (npy_float64 *)PyArray_DATA(dg);

	num_documents = (npy_int) gammas_dims[0];
	num_topics    = (npy_int) gammas_dims[1];
	num_entries   = (npy_int) dwp_dims[0];
	not_done = (int) num_documents;

	long double ttt;
	tot_its = max_its;
	ecnt = num_entries;


	while(not_done > 0 && tot_its != 0){
		tot_its--;

		// For each document compute exp(digamma(gammas_{i,j}) - max_{j}[digamma(gammas_{i,j})])
		for(i = 0; i < num_documents; i++){

			if(isdone_data[i]){
				continue;
			}

			gamma_loc     = gammas_data[i];
			alpha_loc     = alphas_data[i];
			dgamma_loc    = dgammas_data[i];
			rgamma_loc    = rgammas_data[i];
			ttt = gamma_loc[0] + alpha_loc[0];
			dgamma_loc[0] = ld1 = (npy_float64) digammal(ttt);

			for(j = 1; j < num_topics; j++){
				ttt = gamma_loc[j] + alpha_loc[j];
				dgamma_loc[j] = (npy_float64) digammal(ttt);
				if(dgamma_loc[j] > ld1){
					ld1 = dgamma_loc[j];
				}
			}
			for(j = 0; j < num_topics; j++){
				dgamma_loc[j] = exp(dgamma_loc[j] - ld1);
				rgamma_loc[j] = 0;
			}
		}

		// Update counts
		for(i = 0; i < ecnt; i++){

			k          = idxrem_data[i];
			didx_loc   = didx_data[k];

			if(isdone_data[didx_loc]){
				continue;
			}

			dgamma_loc = dgammas_data[didx_loc];
			rgamma_loc = rgammas_data[didx_loc];
			dwp_loc    = dwp_data[k];
			dwc_loc    = dwc_data[k];
			ld1 = 0;
			ww_loc = ww_data[k];
			for(j = 0; j < num_topics; j++){
				ld1 += dwc_loc[j] = dwp_loc[j]*dgamma_loc[j];
			}
			for(j = 0; j < num_topics; j++){
				rgamma_loc[j] += (dwc_loc[j] *= ww_loc/ld1);
			}
		}


		loc_change = 0;
		for(i = 0; i < num_documents; i++){

			if(isdone_data[i]){
				continue;
			}

			alpha_loc = alphas_data[i];
			gamma_loc = gammas_data[i];
			rgamma_loc = rgammas_data[i];
			ld1 = 0;
			ld2 = 0;

			for(j = 0; j < num_topics; j++){
				ld1 += fabs(rgamma_loc[j]-gamma_loc[j]);
				ld2 += gamma_loc[j] + alpha_loc[j];

				gamma_loc[j] = rgamma_loc[j];
			}
			dg_data[i] = (ld1/ld2);

			if(ld1 <= (delta*ld2)){
				isdone_data[i] = 1;
				not_done--;
				loc_change++;
			}
		}


		if(loc_change > 0){
			wcnt = 0;
			for(i = 0; i < ecnt; i++){
				k = idxrem_data[i];
				if(isdone_data[didx_data[k]] == 0){
					idxrem_data[wcnt] = k;
					wcnt++;
				}
			}
			ecnt = wcnt;
		}


	}
	if(tot_its == 0){
		printf("Exceeded maximum iterations. %d of %d remaining.\n", not_done, num_documents);
	}
	PyArray_Free((PyObject *)alphas, (void *) alphas_data);
	PyArray_Free((PyObject *)rgammas, (void *) rgammas_data);
	PyArray_Free((PyObject *)dgammas, (void *) dgammas_data);
	PyArray_Free((PyObject *)gammas, (void *) gammas_data);
	PyArray_Free((PyObject *)dwp, (void *) dwp_data);
	PyArray_Free((PyObject *)dwc, (void *) dwc_data);

 	PyObject *tupleresult = PyTuple_New(2);
    PyTuple_SetItem(tupleresult, 0, PyArray_Return(dwc));
    PyTuple_SetItem(tupleresult, 1, PyArray_Return(gammas));

    return tupleresult;
}



#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "c_ext",
    NULL,
    -1,
    LDAMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_c_ext(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array()
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initc_ext(void)
{
    PyObject *m;

    m = Py_InitModule("c_ext", LDAMethods);
    import_array()
    if (m == NULL) {
        return;
    }
}
#endif