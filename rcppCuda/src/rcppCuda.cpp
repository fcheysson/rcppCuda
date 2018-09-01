#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

extern "C"
void gmatrixMult(double *M, double *N, double *P, int *size);
extern "C"
void amax(double *x, int *index, int *size);
extern "C"
void amax_u(double *x, int *index, int *size);

// [[Rcpp::export]]
int camax_u(arma::vec& x) {
    int size = x.n_elem;
    int index;
	  amax_u(&x, &index, &size);
    return index;
}
