//
// CDDL HEADER START
//
// The contents of this file are subject to the terms of the Common Development
// and Distribution License Version 1.0 (the "License").
//
// You can obtain a copy of the license at
// http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
// specific language governing permissions and limitations under the License.
//
// When distributing Covered Code, include this CDDL HEADER in each file and
// include the License file in a prominent location with the name LICENSE.CDDL.
// If applicable, add the following below this CDDL HEADER, with the fields
// enclosed by brackets "[]" replaced with your own identifying information:
//
// Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
//
// CDDL HEADER END
//

//
// Copyright (c) 2018, Regents of the University of Minnesota.
// All rights reserved.
//
// Contributors:
//    Mingjian Wen
//

#ifndef NETWORK_H_
#define NETWORK_H_

#include "helper.hpp"
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Eigen;

// typedef function pointer
typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
typedef RowMatrixXd (*ActivationFunction)(RowMatrixXd const & x);
typedef RowMatrixXd (*ActivationFunctionDerivative)(RowMatrixXd const & x);

class NeuralNetwork
{
 public:
  NeuralNetwork();
  ~NeuralNetwork();

  void set_nn_structure(int input_size, int num_layers, int * layer_sizes);
  void set_activation(char * name);
  void set_keep_prob(double * keep_prob);
  void add_weight_bias(double ** weight, double * bias, int layer);
  void forward(double * zeta, const int rows, const int cols);
  void backward();

  double get_sum_output() { return activOutputLayer_.sum(); }

  double * get_output() { return activOutputLayer_.data(); }

  double * get_grad_input() { return gradInput_.data(); }

  void echo_input()
  {
    std::cout << "===================================" << std::endl;
    std::cout << "Input data for class NeuralNetwork" << std::endl;
    std::cout << "inputSize_: " << inputSize_ << std::endl;
    std::cout << "Nlayers_: " << Nlayers_ << std::endl;
    std::cout << "Nperceptrons_: ";
    for (size_t i = 0; i < layerSizes_.size(); i++)
    { std::cout << layerSizes_.at(i) << " "; }
    std::cout << std::endl;

    std::cout << "weights and biases:" << std::endl;
    for (size_t i = 0; i < weights_.size(); i++)
    {
      std::cout << "w_" << i << std::endl << weights_.at(i) << std::endl;
      std::cout << "b_" << i << std::endl << biases_.at(i) << std::endl;
    }
  }

 private:
  int inputSize_;  // size of input layer
  int Nlayers_;  // number of layers, including output, excluding input
  std::vector<int> layerSizes_;  // number of perceptrons in each layer
  ActivationFunction activFunc_;
  ActivationFunctionDerivative activFuncDeriv_;
  std::vector<RowMatrixXd> weights_;
  std::vector<RowVectorXd> biases_;
  std::vector<RowMatrixXd> preactiv_;
  std::vector<double> keep_prob_;
  std::vector<RowMatrixXd> keep_prob_binary_;
  RowMatrixXd activOutputLayer_;
  RowMatrixXd gradInput_;

  // dropout
  RowMatrixXd dropout_(RowMatrixXd const & x, int layer);
};

// activation fucntion and derivatives
RowMatrixXd relu(RowMatrixXd const & x);
RowMatrixXd relu_derivative(RowMatrixXd const & x);
RowMatrixXd elu(RowMatrixXd const & x);
RowMatrixXd elu_derivative(RowMatrixXd const & x);
RowMatrixXd tanh(RowMatrixXd const & x);
RowMatrixXd tanh_derivative(RowMatrixXd const & x);
RowMatrixXd sigmoid(RowMatrixXd const & x);
RowMatrixXd sigmoid_derivative(RowMatrixXd const & x);

#endif  // NETWORK_H_
