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


#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include "KIM_ModelDriverHeaders.hpp"
#include "ANN.hpp"
#include "ANNImplementation.hpp"

#define MAXLINE 2048


//==============================================================================
//
// Implementation of ANNImplementation public member functions
//
//==============================================================================

//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

ANNImplementation::ANNImplementation(
  KIM::ModelDriverCreate* const modelDriverCreate,
  KIM::LengthUnit const requestedLengthUnit,
  KIM::EnergyUnit const requestedEnergyUnit,
  KIM::ChargeUnit const requestedChargeUnit,
  KIM::TemperatureUnit const requestedTemperatureUnit,
  KIM::TimeUnit const requestedTimeUnit,
  int* const ier)
  : numberModelSpecies_(0),
  numberUniqueSpeciesPairs_(0),
  cutoff_(NULL),
  lj_A_(0.0),
  lj_cutoff_(0.0),
  cutoffSq_2D_(NULL),
  influenceDistance_(0.0),
  modelWillNotRequestNeighborsOfNoncontributingParticles_(1),
  cachedNumberOfParticles_(0)
{
  // create descriptor and network classes
	descriptor_ = new Descriptor();
	network_ = new NeuralNetwork();


  FILE* parameterFilePointers[MAX_PARAMETER_FILES];
  int numberParameterFiles;

  modelDriverCreate->GetNumberOfParameterFiles(&numberParameterFiles);
  *ier = OpenParameterFiles(modelDriverCreate, numberParameterFiles,
      parameterFilePointers);
  if (*ier) {
    return;
  }

  *ier = ProcessParameterFiles(modelDriverCreate, numberParameterFiles,
      parameterFilePointers);
  CloseParameterFiles(numberParameterFiles, parameterFilePointers);
  if (*ier) {
    return;
  }

  *ier = ConvertUnits(modelDriverCreate,
      requestedLengthUnit,
      requestedEnergyUnit,
      requestedChargeUnit,
      requestedTemperatureUnit,
      requestedTimeUnit);
  if (*ier) {
    return;
  }

  // precompute lookup table
  descriptor_->create_g4_lookup();


  *ier = SetRefreshMutableValues(modelDriverCreate);
  if (*ier) {
    return;
  }

  *ier = RegisterKIMModelSettings(modelDriverCreate);
  if (*ier) {
    return;
  }

// Do not publish parameters
//  *ier = RegisterKIMParameters(modelDriverCreate);
//  if (*ier) {
//    return;
//  }

  *ier = RegisterKIMFunctions(modelDriverCreate);
  if (*ier) {
    return;
  }

  // everything is good
  *ier = false;
  return;
}


//******************************************************************************
ANNImplementation::~ANNImplementation()
{ // note: it is ok to delete a null pointer and we have ensured that
  // everything is initialized to null

  Deallocate1DArray<double>(cutoff_);
  Deallocate2DArray<double>(cutoffSq_2D_);
}


//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelRefresh

int ANNImplementation::Refresh(KIM::ModelRefresh* const modelRefresh)
{
  int ier;

  ier = SetRefreshMutableValues(modelRefresh);
  if (ier) {
    return ier;
  }

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
int ANNImplementation::Compute(
    KIM::ModelCompute const* const modelCompute,
    KIM::ModelComputeArguments const* const modelComputeArguments)
{
  int ier;

  // KIM API Model Input compute flags
  bool isComputeProcess_dEdr = false;
  bool isComputeProcess_d2Edr2 = false;
  //
  // KIM API Model Output compute flags
  bool isComputeEnergy = false;
  bool isComputeForces = false;
  bool isComputeParticleEnergy = false;
  bool isComputeVirial = false;
  bool isComputeParticleVirial = false;
  //
  // KIM API Model Input
  int const* particleSpeciesCodes = NULL;
  int const* particleContributing = NULL;
  VectorOfSizeDIM const* coordinates = NULL;
  //
  // KIM API Model Output
  double* energy = NULL;
  double* particleEnergy = NULL;
  VectorOfSizeDIM* forces = NULL;
  VectorOfSizeSix* virial = NULL;
  VectorOfSizeSix* particleVirial = NULL;

  ier = SetComputeMutableValues(modelComputeArguments,
      isComputeProcess_dEdr, isComputeProcess_d2Edr2,
      isComputeEnergy, isComputeForces, isComputeParticleEnergy,
      isComputeVirial, isComputeParticleVirial,
      particleSpeciesCodes, particleContributing, coordinates,
      energy, forces, particleEnergy, virial, particleVirial);
  if (ier) {
    return ier;
  }

  // Skip this check for efficiency
  //
  //ier = CheckParticleSpecies(modelComputeArguments, particleSpeciesCodes);
  // if (ier) return ier;


#include "ANNImplementationComputeDispatch.cpp"
  return ier;
}


//******************************************************************************
int ANNImplementation::ComputeArgumentsCreate(
    KIM::ModelComputeArgumentsCreate* const modelComputeArgumentsCreate) const
{
  int ier;

  ier = RegisterKIMComputeArgumentsSettings(modelComputeArgumentsCreate);
  if (ier) {
    return ier;
  }

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
int ANNImplementation::ComputeArgumentsDestroy(
    KIM::ModelComputeArgumentsDestroy* const modelComputeArgumentsDestroy)
const
{
  int ier;
  (void) modelComputeArgumentsDestroy; // avoid not used warning

  // nothing else to do for this case

  // everything is good
  ier = false;
  return ier;
}


//==============================================================================
//
// Implementation of ANNImplementation private member functions
//
//==============================================================================

//******************************************************************************
void ANNImplementation::AllocatePrivateParameterMemory()
{
  // nothing to do for this case
}


//******************************************************************************
void ANNImplementation::AllocateParameterMemory()
{ // allocate memory for data
  AllocateAndInitialize1DArray<double> (cutoff_, numberUniqueSpeciesPairs_);
	AllocateAndInitialize2DArray<double> (cutoffSq_2D_, numberModelSpecies_, numberModelSpecies_);
}


//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

int ANNImplementation::OpenParameterFiles(
    KIM::ModelDriverCreate* const modelDriverCreate,
    int const numberParameterFiles,
    FILE* parameterFilePointers[MAX_PARAMETER_FILES])
{
  int ier;

  if (numberParameterFiles > MAX_PARAMETER_FILES) {
    ier = true;
    LOG_ERROR("ANN given too many parameter files");
    return ier;
  }

  for (int i = 0; i < numberParameterFiles; ++i) {
    std::string const* paramFileName;
    ier = modelDriverCreate->GetParameterFileName(i, &paramFileName);
    if (ier) {
      LOG_ERROR("Unable to get parameter file name");
      return ier;
    }

    parameterFilePointers[i] = fopen(paramFileName->c_str(), "r");
    if (parameterFilePointers[i] == 0) {
      char message[MAXLINE];
      sprintf(message,
          "ANN parameter file number %d cannot be opened",
          i);
      ier = true;
      LOG_ERROR(message);
      for (int j = i - 1; i <= 0; --i) {
        fclose(parameterFilePointers[j]);
      }
      return ier;
    }
  }

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

int ANNImplementation::ProcessParameterFiles(
    KIM::ModelDriverCreate* const modelDriverCreate,
    int const numberParameterFiles,
    FILE* const parameterFilePointers[MAX_PARAMETER_FILES])
{

//  int N, ier;
//  int endOfFileFlag = 0;
//  char spec1[MAXLINE], spec2[MAXLINE], nextLine[MAXLINE];
//  int iIndex, jIndex, indx;
//  double next_A, next_B, next_p, next_q, next_sigma, next_lambda, next_gamma;
//  double next_costheta0, next_cutoff;
//
//  (void) numberParameterFiles; // avoid not used warning
//
//  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
//  ier = sscanf(nextLine, "%d", &N);
//  if (ier != 1) {
//    sprintf(nextLine, "unable to read first line of the parameter file");
//    ier = true;
//    LOG_ERROR(nextLine);
//    fclose(parameterFilePointers[0]);
//    return ier;
//  }
//  numberModelSpecies_ = N;
//  numberUniqueSpeciesPairs_ = ((numberModelSpecies_ + 1) * numberModelSpecies_) / 2;
//  AllocateParameterMemory();
//
//  // set all values of p_ to -1.1e10 for later check that we have read all params
//  for (int i = 0; i < ((N + 1) * N / 2); i++) {
//    p_[i] = -1.1e10;
//  }
//
//  // keep track of known species
//  std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator> modelSpeciesMap;
//  int index = 0;   // species code integer code starting from 0
//
//  // Read and process data lines
//  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
//  while (endOfFileFlag == 0)
//  {
//    ier = sscanf(nextLine, "%s %s %lg %lg %lg %lg %lg %lg %lg %lg %lg",
//        spec1, spec2, &next_A, &next_B, &next_p, &next_q, &next_sigma,
//        &next_lambda, &next_gamma, &next_costheta0, &next_cutoff);
//    if (ier != 11) {
//      sprintf(nextLine, "error reading lines of the parameter file");
//      LOG_ERROR(nextLine);
//      return true;
//    }
//
//    // convert species strings to proper type instances
//    KIM::SpeciesName const specName1(spec1);
//    KIM::SpeciesName const specName2(spec2);
//     if ((specName1.String() == "unknown") ||
//         (specName2.String() == "unknown") ) {
//      sprintf(nextLine, "error parameter file: get unknown species");
//      LOG_ERROR(nextLine);
//      return true;
//    }
//
//
//    // check for new species
//    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>::
//    const_iterator iIter = modelSpeciesMap.find(specName1);
//    if (iIter == modelSpeciesMap.end()) {
//      modelSpeciesMap[specName1] = index;
//      modelSpeciesCodeList_.push_back(index);
//
//      ier = modelDriverCreate->SetSpeciesCode(specName1, index);
//      if (ier) {
//        return ier;
//      }
//      iIndex = index;
//      index++;
//    }
//    else {
//      iIndex = modelSpeciesMap[specName1];
//    }
//
//    std::map<KIM::SpeciesName const, int, KIM::SPECIES_NAME::Comparator>::
//    const_iterator jIter = modelSpeciesMap.find(specName2);
//    if (jIter == modelSpeciesMap.end()) {
//      modelSpeciesMap[specName2] = index;
//      modelSpeciesCodeList_.push_back(index);
//
//      ier = modelDriverCreate->SetSpeciesCode(specName2, index);
//      if (ier) {
//        return ier;
//      }
//      jIndex = index;
//      index++;
//    }
//    else {
//      jIndex = modelSpeciesMap[specName2];
//    }
//
//    if (iIndex >= jIndex) {
//      indx = jIndex * N + iIndex - (jIndex * jIndex + jIndex) / 2;
//    }
//    else {
//      indx = iIndex * N + jIndex - (iIndex * iIndex + iIndex) / 2;
//    }
//    A_[indx] = next_A;
//    B_[indx] = next_B;
//    p_[indx] = next_p;
//    q_[indx] = next_q;
//    sigma_[indx] = next_sigma;
//    lambda_[indx] = next_lambda;
//    gamma_[indx] = next_gamma;
//    costheta0_[indx] = next_costheta0;
//    cutoff_[indx] = next_cutoff;
//
//    getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
//  }
//
//  // check we have read all parameters
//  for (int i = 0; i < ((N + 1) * N / 2); i++) {
//    if (p_[i] < -1e10) {
//      sprintf(nextLine, "error: not enough parameter data.\n");
//      sprintf(nextLine, "%d species requires %d data lines.", N, (N + 1) * N / 2);
//      LOG_ERROR(nextLine);
//      return true;
//    }
//  }
//



  int ier;
  int index;
  char spec[MAXLINE];

  //int N;
  int endOfFileFlag = 0;
  char nextLine[MAXLINE];
  char errorMsg[MAXLINE];
  char name[MAXLINE];
	double cutoff;

  // descriptor
	int numDescTypes;
	int numDescs;
	int numParams;
	int numParamSets;
	double** descParams = NULL;

  // network
  int numLayers;
  int* numPerceptrons;



  // lj part
  getNextDataLine(parameterFilePointers[1], nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%s %lf %lf", spec, &lj_A_, &lj_cutoff_);
  if (ier != 3) {
    sprintf(errorMsg, "unable to lj parameters from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  index = 0;
  KIM::SpeciesName const specName(spec);
  ier = modelDriverCreate->SetSpeciesCode(specName, index);
  if (ier) {
    return ier;
  }
  modelSpeciesCodeList_.push_back(index);

  numberModelSpecies_ = 1;
  numberUniqueSpeciesPairs_ = ((numberModelSpecies_ + 1) * numberModelSpecies_) / 2;
  AllocateParameterMemory();


  // NN part
	// cutoff
  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%s %lf", name, &cutoff);
  if (ier != 2) {
    sprintf(errorMsg, "unable to read cutoff from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // register cutoff
  lowerCase(name);
  if (strcmp(name, "cos") != 0 && strcmp(name, "exp") != 0)
  {
    sprintf(errorMsg, "unsupported cutoff type. Expecting `cos', or `exp' "
        "given %s.\n", name);
    LOG_ERROR(errorMsg);
    return true;
  }
	descriptor_->set_cutfunc(name);

//TODO modifiy this such that each pair has its own cutoff
// use of numberUniqueSpeciesPairs is not good. Since it requires the Model
// provide all the params that the Driver supports. number of species should
// be read in from the input file.
  for (int i=0; i<numberUniqueSpeciesPairs_; i++) {
	  cutoff_[i] = cutoff;
  }

	// number of descriptor types
  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%d", &numDescTypes);
  if (ier != 1) {
    sprintf(errorMsg, "unable to read number of descriptor types from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // descriptor
  for (int i=0; i<numDescTypes; i++) {
    // descriptor name and parameter dimensions
    getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);

    // name of descriptor
    ier = sscanf(nextLine, "%s", name);
    if (ier != 1) {
      sprintf(errorMsg, "unable to read descriptor from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }
    lowerCase(name); // change to lower case name
    if (strcmp(name, "g1") == 0) {  // G1
      descriptor_->add_descriptor(name, NULL, 1, 0);
    }
    else{
      // re-read name, and read number of param sets and number of params
      ier = sscanf(nextLine, "%s %d %d", name, &numParamSets, &numParams);
      if (ier != 3) {
        sprintf(errorMsg, "unable to read descriptor from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
      // change name to lower case
      lowerCase(name);

      // check size of params is correct w.r.t its name
      if (strcmp(name, "g2") == 0) {
        if (numParams != 2) {
          sprintf(errorMsg, "number of params for descriptor G2 is incorrect, "
              "expecting 2, but given %d.\n", numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g3") == 0) {
        if (numParams != 1) {
          sprintf(errorMsg, "number of params for descriptor G3 is incorrect, "
              "expecting 1, but given %d.\n", numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g4") == 0) {
        if (numParams != 3) {
          sprintf(errorMsg, "number of params for descriptor G4 is incorrect, "
              "expecting 3, but given %d.\n", numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else if (strcmp(name, "g5") == 0) {
        if (numParams != 3) {
          sprintf(errorMsg, "number of params for descriptor G5 is incorrect, "
              "expecting 3, but given %d.\n", numParams);
          LOG_ERROR(errorMsg);
          return true;
        }
      }
      else {
        sprintf(errorMsg, "unsupported descriptor `%s' from line:\n", name);
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }

      // read descriptor params
      AllocateAndInitialize2DArray<double> (descParams, numParamSets, numParams);
      for (int j=0; j<numParamSets; j++) {
        getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
        ier = getXdouble(nextLine, numParams, descParams[j]);
        if (ier) {
          sprintf(errorMsg, "unable to read descriptor parameters from line:\n");
          strcat(errorMsg, nextLine);
          LOG_ERROR(errorMsg);
          return true;
        }
      }

      // copy data to Descriptor
      descriptor_->add_descriptor(name, descParams, numParamSets, numParams);
      Deallocate2DArray(descParams);
    }
  }
  // number of descriptors
  numDescs = descriptor_->get_num_descriptors();


  // centering and normalizing params
  // flag, whether we use this feature
  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%*s %s", name);
  if (ier != 1) {
    sprintf(errorMsg, "unable to read centering and normalization info from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  lowerCase(name);
  bool do_center_and_normalize;
  if (strcmp(name, "true") == 0) {
    do_center_and_normalize = true;
  } else {
    do_center_and_normalize = false;
  }

  int size=0;
  double* means = NULL;
  double* stds = NULL;
  if (do_center_and_normalize)
  {
    // size of the data, this should be equal to numDescs
    getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
    ier = sscanf(nextLine, "%d", &size);
    if (ier != 1) {
      sprintf(errorMsg, "unable to read the size of centering and normalization "
          "data info from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }
    if (size != numDescs) {
      sprintf(errorMsg, "Size of centering and normalizing data inconsistent with "
          "the number of descriptors. Size = %d, num_descriptors=%d\n", size, numDescs);
      LOG_ERROR(errorMsg);
      return true;
    }

    // read means
    AllocateAndInitialize1DArray<double> (means, size);
    for (int i=0; i<size; i++) {
      getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
      ier = sscanf(nextLine, "%lf", &means[i]);
      if (ier != 1) {
        sprintf(errorMsg, "unable to read `means' from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }

    // read standard deviations
    AllocateAndInitialize1DArray<double> (stds, size);
    for (int i=0; i<size; i++) {
      getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
      ier = sscanf(nextLine, "%lf", &stds[i]);
      if (ier != 1) {
        sprintf(errorMsg, "unable to read `means' from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }
  }

  // store info into descriptor class
	descriptor_->set_center_and_normalize(do_center_and_normalize, size, means, stds);
  Deallocate1DArray(means);
  Deallocate1DArray(stds);


//TODO delete
//  descriptor_->echo_input();


  // network structure
  // number of layers
  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%d", &numLayers);
  if (ier != 1) {
    sprintf(errorMsg, "unable to read number of layers from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // number of perceptrons in each layer
  numPerceptrons = new int[numLayers];
  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = getXint(nextLine, numLayers, numPerceptrons);
  if (ier) {
    sprintf(errorMsg, "unable to read number of perceptrons from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // copy to network class
  network_->set_nn_structure(numDescs, numLayers, numPerceptrons);


  // activation function
  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = sscanf(nextLine, "%s", name);
  if (ier != 1) {
    sprintf(errorMsg, "unable to read `activation function` from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }

  // register activation function
  lowerCase(name);
  if (strcmp(name, "sigmoid") != 0
      && strcmp(name, "tanh") != 0
      && strcmp(name, "relu") != 0
      && strcmp(name, "elu") != 0)
  {
    sprintf(errorMsg, "unsupported activation function. Expecting `sigmoid`, `tanh` "
        " `relu` or `elu`, given %s.\n", name);
    LOG_ERROR(errorMsg);
    return true;
  }
  network_->set_activation(name);


  // keep probability
  double* keep_prob;
  AllocateAndInitialize1DArray<double> (keep_prob, numLayers);

  getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
  ier = getXdouble(nextLine, numLayers, keep_prob);
  if (ier) {
    sprintf(errorMsg, "unable to read `keep probability` from line:\n");
    strcat(errorMsg, nextLine);
    LOG_ERROR(errorMsg);
    return true;
  }
  network_->set_keep_prob(keep_prob);
  Deallocate1DArray(keep_prob);


  // weights and biases
  for (int i=0; i<numLayers; i++) {

    double** weight;
	  double* bias;
    int row;
    int col;

    if (i==0) {
      row = numDescs;
      col = numPerceptrons[i];
    } else {
      row = numPerceptrons[i-1];
      col = numPerceptrons[i];
    }

    AllocateAndInitialize2DArray<double> (weight, row, col);
    for (int j=0; j<row; j++) {
      getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
      ier = getXdouble(nextLine, col, weight[j]);
      if (ier) {
        sprintf(errorMsg, "unable to read `weight` from line:\n");
        strcat(errorMsg, nextLine);
        LOG_ERROR(errorMsg);
        return true;
      }
    }

    // bias
    AllocateAndInitialize1DArray<double> (bias, col);
    getNextDataLine(parameterFilePointers[0], nextLine, MAXLINE, &endOfFileFlag);
    ier = getXdouble(nextLine, col, bias);
    if (ier) {
      sprintf(errorMsg, "unable to read `bias` from line:\n");
      strcat(errorMsg, nextLine);
      LOG_ERROR(errorMsg);
      return true;
    }

    // copy to network class
    network_->add_weight_bias(weight, bias, i);

    Deallocate2DArray(weight);
    Deallocate1DArray(bias);
  }

  delete [] numPerceptrons;


//TODO delete
//  network_->echo_input();


  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
void ANNImplementation::getNextDataLine(
    FILE* const filePtr, char* nextLinePtr, int const maxSize,
    int* endOfFileFlag)
{
  do
  {
    if (fgets(nextLinePtr, maxSize, filePtr) == NULL) {
      *endOfFileFlag = 1;
      break;
    }

    while ((nextLinePtr[0] == ' ' || nextLinePtr[0] == '\t') ||
        (nextLinePtr[0] == '\n' || nextLinePtr[0] == '\r'))
    {
      nextLinePtr = (nextLinePtr + 1);
    }
  } while ((strncmp("#", nextLinePtr, 1) == 0) || (strlen(nextLinePtr) == 0));

  // remove comments starting with `#' in a line
  char* pch = strchr(nextLinePtr, '#');
  if (pch != NULL) {
    *pch = '\0';
  }

}

//******************************************************************************
int ANNImplementation::getXdouble(char* linePtr, const int N, double* list)
{
  int ier;
  char * pch;
  char line[MAXLINE];
  int i = 0;

  strcpy(line, linePtr);
  pch = strtok(line, " \t\n\r");
  while (pch != NULL) {
    ier = sscanf(pch, "%lf", &list[i]);
    if (ier != 1) {
      return true;
    }
    pch = strtok(NULL, " \t\n\r");
    i += 1;
  }

  if (i != N) {
    return true;
  }

  return false;
}

//******************************************************************************
int ANNImplementation::getXint(char* linePtr, const int N, int* list)
{
  int ier;
  char * pch;
  char line[MAXLINE];
  int i = 0;

  strcpy(line, linePtr);
  pch = strtok(line, " \t\n\r");
  while (pch != NULL) {
    ier = sscanf(pch, "%d", &list[i]);
    if (ier != 1) {
      return true;
    }
    pch = strtok(NULL, " \t\n\r");
    i += 1;
  }
  if (i != N) {
    return true;
  }

  return false;
}

//******************************************************************************
void ANNImplementation::lowerCase(char* linePtr)
{
  for(int i=0; linePtr[i]; i++){
    linePtr[i] = tolower(linePtr[i]);
  }
}

//******************************************************************************
void ANNImplementation::CloseParameterFiles(
    int const numberParameterFiles,
    FILE* const parameterFilePointers[MAX_PARAMETER_FILES])
{
  for (int i = 0; i < numberParameterFiles; ++i) {
    fclose(parameterFilePointers[i]);
  }
}


//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

int ANNImplementation::ConvertUnits(
    KIM::ModelDriverCreate* const modelDriverCreate,
    KIM::LengthUnit const requestedLengthUnit,
    KIM::EnergyUnit const requestedEnergyUnit,
    KIM::ChargeUnit const requestedChargeUnit,
    KIM::TemperatureUnit const requestedTemperatureUnit,
    KIM::TimeUnit const requestedTimeUnit)
{
  int ier;

  // define default base units
  KIM::LengthUnit fromLength = KIM::LENGTH_UNIT::A;
  KIM::EnergyUnit fromEnergy = KIM::ENERGY_UNIT::eV;
  KIM::ChargeUnit fromCharge = KIM::CHARGE_UNIT::e;
  KIM::TemperatureUnit fromTemperature = KIM::TEMPERATURE_UNIT::K;
  KIM::TimeUnit fromTime = KIM::TIME_UNIT::ps;

  // changing units of sigma, gamma, and cutoff
  double convertLength = 1.0;

  ier = modelDriverCreate->ConvertUnit(
      fromLength, fromEnergy, fromCharge, fromTemperature, fromTime,
      requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
      requestedTemperatureUnit, requestedTimeUnit,
      1.0, 0.0, 0.0, 0.0, 0.0,
      &convertLength);
  if (ier) {
    LOG_ERROR("Unable to convert length unit");
    return ier;
  }
  // convert to active units
  if (convertLength != ONE) {
    //for (int i = 0; i < numberUniqueSpeciesPairs_; ++i) {
    //}
    lj_cutoff_ *= convertLength;
  }

  // changing units of A and lambda
  double convertEnergy = 1.0;
  ier = modelDriverCreate->ConvertUnit(
      fromLength, fromEnergy, fromCharge, fromTemperature, fromTime,
      requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
      requestedTemperatureUnit, requestedTimeUnit,
      0.0, 1.0, 0.0, 0.0, 0.0,
      &convertEnergy);
  if (ier) {
    LOG_ERROR("Unable to convert energy unit");
    return ier;
  }
  // convert to active units
  if (convertEnergy != ONE) {
    //for (int i = 0; i < numberUniqueSpeciesPairs_; ++i) {
    //}
      lj_A_ *= convertEnergy;
  }

  // register units
  ier = modelDriverCreate->SetUnits(
      requestedLengthUnit,
      requestedEnergyUnit,
      KIM::CHARGE_UNIT::unused,
      KIM::TEMPERATURE_UNIT::unused,
      KIM::TIME_UNIT::unused);
  if (ier) {
    LOG_ERROR("Unable to set units to requested values");
    return ier;
  }

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
int ANNImplementation::RegisterKIMModelSettings(
    KIM::ModelDriverCreate* const modelDriverCreate) const
{
  // register numbering
  int error = modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased);

  return error;
}


//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArgumentsCreate

int ANNImplementation::RegisterKIMComputeArgumentsSettings(
    KIM::ModelComputeArgumentsCreate* const modelComputeArgumentsCreate) const
{
  // register arguments
  LOG_INFORMATION("Register argument supportStatus");

  int error =
    modelComputeArgumentsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
        KIM::SUPPORT_STATUS::optional) ||
    modelComputeArgumentsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialForces,
        KIM::SUPPORT_STATUS::optional) ||
    modelComputeArgumentsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy,
        KIM::SUPPORT_STATUS::optional) ||
    modelComputeArgumentsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
        KIM::SUPPORT_STATUS::optional) ||
    modelComputeArgumentsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialParticleVirial,
        KIM::SUPPORT_STATUS::optional);

  // register callbacks
  LOG_INFORMATION("Register callback supportStatus");
  error =
    error ||
    modelComputeArgumentsCreate->SetCallbackSupportStatus(
        KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
        KIM::SUPPORT_STATUS::optional) ||
    modelComputeArgumentsCreate->SetCallbackSupportStatus(
        KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
        KIM::SUPPORT_STATUS::optional);

  return error;
}


//******************************************************************************
// helper macro
#define SNUM( x  ) static_cast<std::ostringstream &>(    \
    std::ostringstream() << std::dec << x).str()

#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelDriverCreate

int ANNImplementation::RegisterKIMParameters(
    KIM::ModelDriverCreate* const modelDriverCreate)
{
  // Do not support the publish of parameters

  // everything is good
  int ier = false;
  return ier;
}


//******************************************************************************
int ANNImplementation::RegisterKIMFunctions(
    KIM::ModelDriverCreate* const modelDriverCreate)
const
{
  int error;

  // register the Destroy(), Refresh(), and Compute() functions
  error =
    modelDriverCreate->SetDestroyPointer(
        KIM::LANGUAGE_NAME::cpp,
        (KIM::Function*)&(ANN::Destroy)) ||
// Do not publish parameters
//    modelDriverCreate->SetRefreshPointer(
//        KIM::LANGUAGE_NAME::cpp,
//        (KIM::Function*)&(ANN::Refresh)) ||
    modelDriverCreate->SetComputePointer(
        KIM::LANGUAGE_NAME::cpp,
        (KIM::Function*)&(ANN::Compute)) ||
    modelDriverCreate->SetComputeArgumentsCreatePointer(
        KIM::LANGUAGE_NAME::cpp,
        (KIM::Function*)&(ANN::ComputeArgumentsCreate)) ||
    modelDriverCreate->SetComputeArgumentsDestroyPointer(
        KIM::LANGUAGE_NAME::cpp,
        (KIM::Function*)&(ANN::ComputeArgumentsDestroy));

  return error;
}


//******************************************************************************
template<class ModelObj>
int ANNImplementation::SetRefreshMutableValues(
    ModelObj* const modelObj)
{ // use (possibly) new values of parameters to compute other quantities
  // NOTE: This function is templated because it's called with both a
  //       modelDriverCreate object during initialization and with a
  //       modelRefresh object when the Model's parameters have been altered
  int ier;

  // update parameters
  for (int i = 0; i < numberModelSpecies_; ++i) {
    for (int j = 0; j <= i; ++j) {
      int const index = j * numberModelSpecies_ + i - (j * j + j) / 2;
      cutoffSq_2D_[i][j] = cutoffSq_2D_[j][i] = cutoff_[index] * cutoff_[index];
    }
  }

  // update cutoff value in KIM API object
  influenceDistance_ = 0.0;

  for (int i = 0; i < numberModelSpecies_; i++) {
    int indexI = modelSpeciesCodeList_[i];

    for (int j = 0; j < numberModelSpecies_; j++) {
      int indexJ = modelSpeciesCodeList_[j];

      if (influenceDistance_ < cutoffSq_2D_[indexI][indexJ]) {
        influenceDistance_ = cutoffSq_2D_[indexI][indexJ];
      }
    }
  }

  influenceDistance_ = sqrt(influenceDistance_);

  // compare with lj cutoff

  if(influenceDistance_ < lj_cutoff_) {
    influenceDistance_ = lj_cutoff_;
  }

  modelObj->SetInfluenceDistancePointer(&influenceDistance_);
  modelObj->SetNeighborListPointers(1,
      &influenceDistance_, &modelWillNotRequestNeighborsOfNoncontributingParticles_);

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelComputeArguments

int ANNImplementation::SetComputeMutableValues(
    KIM::ModelComputeArguments const* const modelComputeArguments,
    bool& isComputeProcess_dEdr,
    bool& isComputeProcess_d2Edr2,
    bool& isComputeEnergy,
    bool& isComputeForces,
    bool& isComputeParticleEnergy,
    bool& isComputeVirial,
    bool& isComputeParticleVirial,
    int const*& particleSpeciesCodes,
    int const*& particleContributing,
    VectorOfSizeDIM const*& coordinates,
    double*& energy,
    VectorOfSizeDIM*& forces,
    double*& particleEnergy,
    VectorOfSizeSix*& virial,
    VectorOfSizeSix*& particleVirial)
{
  int ier = true;

  // get compute flags
  int compProcess_dEdr;
  int compProcess_d2Edr2;

  modelComputeArguments->IsCallbackPresent(
      KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
      &compProcess_dEdr);
  modelComputeArguments->IsCallbackPresent(
      KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
      &compProcess_d2Edr2);

  isComputeProcess_dEdr = compProcess_dEdr;
  isComputeProcess_d2Edr2 = compProcess_d2Edr2;

  int const* numberOfParticles;
  ier =
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
        &numberOfParticles) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
        &particleSpeciesCodes) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
        &particleContributing) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::coordinates,
        (double const** const)&coordinates) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
        &energy) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialForces,
        (double const** const)&forces) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialParticleEnergy,
        &particleEnergy) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
        (double const** const)&virial) ||
    modelComputeArguments->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialParticleVirial,
        (double const** const)&particleVirial);
  if (ier) {
    LOG_ERROR("GetArgumentPointer");
    return ier;
  }

  isComputeEnergy = (energy != NULL);
  isComputeForces = (forces != NULL);
  isComputeParticleEnergy = (particleEnergy != NULL);
  isComputeVirial = (virial != NULL);
  isComputeParticleVirial = (particleVirial != NULL);

  // update values
  cachedNumberOfParticles_ = *numberOfParticles;

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
// Assume that the particle species interge code starts from 0
#undef  KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCompute

int ANNImplementation::CheckParticleSpeciesCodes(
    KIM::ModelCompute const* const modelCompute,
    int const* const particleSpeciesCodes) const
{
  int ier;

  for (int i = 0; i < cachedNumberOfParticles_; ++i) {
    if ((particleSpeciesCodes[i] < 0) || (particleSpeciesCodes[i] >= numberModelSpecies_)) {
      ier = true;
      LOG_ERROR("unsupported particle species codes detected");
      return ier;
    }
  }

  // everything is good
  ier = false;
  return ier;
}


//******************************************************************************
int ANNImplementation::GetComputeIndex(
    const bool& isComputeProcess_dEdr,
    const bool& isComputeProcess_d2Edr2,
    const bool& isComputeEnergy,
    const bool& isComputeForces,
    const bool& isComputeParticleEnergy,
    const bool& isComputeVirial,
    const bool& isComputeParticleVirial) const
{
  //const int processdE = 2;
  const int processd2E = 2;
  const int energy = 2;
  const int force = 2;
  const int particleEnergy = 2;
  const int virial = 2;
  const int particleVirial = 2;


  int index = 0;

  // processdE
  index += (int(isComputeProcess_dEdr))
           * processd2E * energy * force * particleEnergy * virial * particleVirial;

  // processd2E
  index += (int(isComputeProcess_d2Edr2))
           * energy * force * particleEnergy * virial * particleVirial;

  // energy
  index += (int(isComputeEnergy))
           * force * particleEnergy * virial * particleVirial;

  // force
  index += (int(isComputeForces))
           * particleEnergy * virial * particleVirial;

  // particleEnergy
  index += (int(isComputeParticleEnergy))
           * virial * particleVirial;

  // virial
  index += (int(isComputeVirial))
           * particleVirial;

  // particleVirial
  index += (int(isComputeParticleVirial));

  return index;
}


//==============================================================================
//
// LJ functions
//
//==============================================================================

void ANNImplementation::calc_phi(double const epsilon, double const sigma,
    double const cutoff, double const r, double * const phi) const
{

  double sor, sor6, sor12;

  if (r >= cutoff) {
    *phi = 0;
  }
  else {
    sor  = sigma/r;
    sor6 = sor*sor*sor;
    sor6 = sor6*sor6;
    //sor12= sor6*sor6;
    sor12= 0;
    *phi = 4.0*epsilon*(sor12-sor6);
  }

}


void ANNImplementation::calc_phi_dphi(double const epsilon, double const sigma,
    double const cutoff, double const r, double * const phi, double * const dphi) const
{
  double sor, sor6, sor12;

  if (r >= cutoff) {
    *phi = 0;
    *dphi = 0;
  }
  else {
    sor  = sigma/r;
    sor6 = sor*sor*sor;
    sor6 = sor6*sor6;
    //sor12= sor6*sor6;
    sor12= 0;
    *phi = 4.0*epsilon*(sor12-sor6);
    *dphi = 24.0*epsilon*(-2.0*sor12 + sor6)/r;
  }

}


/* switch function  */
void ANNImplementation::switch_fn(double const x_min, double const x_max,
    double const x, double *const fn, double * const fn_prime) const
{
  double t;
  double t_sq;
  double t_cubic;

  if (x <= x_min) {
    *fn = 1;
    *fn_prime = 0;
  }
  else if (x >= x_max) {
    *fn = 0;
    *fn_prime = 0;
  }
  else {
    t = (x - x_min)/(x_max - x_min);
    t_sq = t*t;
    t_cubic = t_sq*t;
    *fn = t_cubic*(-10.0 +15*t -6*t_sq) + 1;
    *fn_prime = t_sq*(-30 + 60*t - 30*t_sq)/(x_max-x_min);
  }

}

