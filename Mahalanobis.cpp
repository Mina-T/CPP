#include <iostream>
#include <cmath>
#include "csv.h"
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cnpy.h> 
#include <Eigen/Dense>
using namespace Eigen; 
using namespace std;
// How close a point actually is to a distribution of points, a multivariate equivalent of the Euclidean distance
// g++ -o Mahalanobis Mahalanobis.cpp -pthread -I/leonardo/home/userexternal/mtaleblo/cnpy -L/leonardo/home/userexternal/mtaleblo/cnpy/build -lcnpy//

struct Columns {
    vector<float> ref;
    vector<float> avg;
    vector<float> std_dev;
};

// load CSV file, returns ref, avg, std of the ensembels
Columns load_csv(string filename){
	io::CSVReader<3> in(filename);
	in.read_header(io::ignore_extra_column, "ref", "avg", "std");
	//, "c1", "c2", "c3","c4","c5","c6","c7","c8","c9","c10");//
	float ref, avg, std_dev;
	Columns cols;

	while (in.read_row(ref, avg, std_dev)){
	cols.ref.push_back(ref);
        cols.avg.push_back(avg);
        cols.std_dev.push_back(std_dev);
	}

	return cols;

}

// compute calibration factor
float alpha_sqr(const vector<float> &ref, const vector<float> &avg, const vector<float> &std_dev){
        size_t ref_size = ref.size();
	vector <float> diff(ref_size);
	transform(ref.begin(), ref.end(), avg.begin(), diff.begin(), minus <float> ());
	vector <float> squared_diff(ref_size);
        transform(diff.begin(), diff.end(), squared_diff.begin(), [](float d){
			return d * d;
			});
	vector <float> squared_std(ref_size);
	transform(std_dev.begin(), std_dev.end(), squared_std.begin(), [](float d){
                        return d * d;
                        });
        vector <float> division(ref_size);
        transform(squared_diff.begin(), squared_diff.end(), squared_std.begin(), division.begin(), divides<float> ());
	float sum_ = accumulate(division.begin(), division.end(), 0.0f);
	float alpha_2 = sum_ / static_cast<float>(ref.size());
	cout << "alpha squared is: " << alpha_2 << "\n";
	return alpha_2;
}


// load the input file
vector<vector<double>> load_npy_file(const string &filename){
	try {
		cnpy::NpyArray arr = cnpy::npy_load(filename);
	 
	size_t n_atom = arr.shape[0];
	size_t n_features = arr.shape[1];
	cout << "input array dimension: " << n_atom << "*" << n_features << "\n";
	double* data = arr.data<double>();
	size_t n_rows_to_load = static_cast<size_t>(100);
	vector<vector<double>> F_matrix(n_rows_to_load, vector<double>(n_features));
        for (size_t i = 0; i < n_rows_to_load; i++) {
        for (size_t  j = 0; j < n_features; j ++ ) {
		F_matrix[i][j] = data[i * n_features + j];
	}
	}
	cout << "F_matrix.shape = " << F_matrix.size() << "*" << F_matrix[0].size() << "\n";
	return F_matrix;
	}
	catch (const exception &e)  {
                cerr << "Error loading file: " << e.what() << endl;
                return {};
        }

}



// compute the covariance of the training set (or any reference set)
vector<vector<double>> covariance(const vector<vector<double>> &F_matrix){
	size_t n_atom = F_matrix.size();
        size_t n_features = F_matrix[0].size();
        // compute the mean of the dataset
	vector<double> means(n_features, 0);
	for (size_t j = 0; j < n_features; j ++){
	means[j] = 0;
	    for( size_t i = 0 ; i < n_atom ; i ++){
	    means[j] += F_matrix[i][j];}
	means[j] /= n_atom ;
	}
        // center the dataset
        vector<vector<double>> centered_features(n_atom, vector<double>(n_features));
	for (size_t i = 0; i < n_atom; i ++){
		for (size_t j = 0; j < n_features; j ++){
			centered_features[i][j] = F_matrix[i][j] - means[j];
		}
	}
        // compute the covariance of the datset
	vector<vector<double>> cov_matrix(n_features, vector<double>(n_features));
	for (size_t j1 = 0; j1 < n_features; j1 ++){
	    for (size_t j2 = 0; j2 < n_features; j2 ++) {
		    double sum_cov = 0;
		    for (size_t i = 0; i < n_atom; i++ ){
			    sum_cov += centered_features[i][j1] * centered_features[i][j2];
		    }
		    sum_cov /= n_atom - 1;
		    cov_matrix[j1][j2] = sum_cov;
	    }
	}	
return cov_matrix ;
}

// inverse the covariance matrix
vector<vector<double>> inverse(const vector<vector<double>> &cov_matrix){
	if (cov_matrix.size() != cov_matrix[0].size()) {
        throw runtime_error("Matrix must be square to invert.");
    }
	// convert to eigen matrix type
	MatrixXd eigen_cov_matrix(cov_matrix.size(), cov_matrix[0].size());
	for (size_t i = 0; i < cov_matrix.size(); ++i) {
            for (size_t j = 0; j < cov_matrix[0].size(); ++j) {
                eigen_cov_matrix(i, j) = cov_matrix[i][j];
        }
    }
	MatrixXd eigen_inv_matrix;
        FullPivLU<MatrixXd> lu_decomp(eigen_cov_matrix);
        
	if (!lu_decomp.isInvertible()) {
        cout << "Covariance matrix is singular, using ridge regularization:" << "\n";
	double epsilon = 1e-8;
       	// add a regularization value (epsilon) in case of a Singular matrix, compute the inverse
        MatrixXd regularized = eigen_cov_matrix + epsilon * MatrixXd::Identity(eigen_cov_matrix.rows(), eigen_cov_matrix.cols());
        eigen_inv_matrix =  regularized.inverse();
	}
	
	else {
	MatrixXd eigen_inv_matrix = eigen_cov_matrix.inverse();	
	}
        // convert the inversed matrix from eigen matrix to vector type
	vector<vector<double>> result(eigen_inv_matrix.rows(), vector<double>(eigen_inv_matrix.cols()));
        for (size_t i = 0; i < eigen_inv_matrix.rows(); ++i) {
            for (size_t j = 0; j < eigen_inv_matrix.cols(); ++j) {
                result[i][j] = eigen_inv_matrix(i, j);
	    }
	}
        return result;
}


int main(char* argv[]){
        string filename = argv[1];
	//cin >> filename ;
	cout << "reading from" << filename << "\n";
	vector<vector<double>>  F_matrix = load_npy_file(filename);
	cout << F_matrix[0][0] <<"\n";
	vector<vector<double>>  cov_matrix = covariance(F_matrix);
	cout << cov_matrix[0][0] <<"\n";
	vector<vector<double>> inversed = inverse(cov_matrix);
	cout << inversed[0][0] <<"\n";

	return 0;

}

