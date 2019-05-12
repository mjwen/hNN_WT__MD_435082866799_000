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

#ifndef ANN_IMPLEMENTATION_HPP_
#define ANN_IMPLEMENTATION_HPP_

#include <vector>
#include "KIM_LogMacros.hpp"
#include "ANN.hpp"
#include "descriptor.h"
#include "helper.hpp"
#include "network.h"

#define DIM 3
#define ONE 1.0

#define MAX_PARAMETER_FILES 2

//==============================================================================
//
// Declaration of ANNImplementation class
//
//==============================================================================

//******************************************************************************
class ANNImplementation
{
 public:
  ANNImplementation(KIM::ModelDriverCreate * const modelDriverCreate,
                    KIM::LengthUnit const requestedLengthUnit,
                    KIM::EnergyUnit const requestedEnergyUnit,
                    KIM::ChargeUnit const requestedChargeUnit,
                    KIM::TemperatureUnit const requestedTemperatureUnit,
                    KIM::TimeUnit const requestedTimeUnit,
                    int * const ier);
  ~ANNImplementation();  // no explicit Destroy() needed here

  int Refresh(KIM::ModelRefresh * const modelRefresh);
  int Compute(KIM::ModelCompute const * const modelCompute,
              KIM::ModelComputeArguments const * const modelComputeArguments);
  int ComputeArgumentsCreate(KIM::ModelComputeArgumentsCreate * const
                                 modelComputeArgumentsCreate) const;
  int ComputeArgumentsDestroy(KIM::ModelComputeArgumentsDestroy * const
                                  modelComputeArgumentsDestroy) const;

 private:
  // Constant values that never change
  //   Set in constructor (via SetConstantValues)
  //
  //
  // ANNImplementation: constants
  int numberModelSpecies_;
  std::vector<int> modelSpeciesCodeList_;
  int numberUniqueSpeciesPairs_;

  // Constant values that are read from the input files and never change
  //   Set in constructor (via functions listed below)
  //
  //
  // Private Model Parameters
  //   Memory allocated in AllocatePrivateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines
  // none
  //
  // KIM API: Model Parameters whose (pointer) values never change
  //   Memory allocated in AllocateParameterMemory() (from constructor)
  //   Memory deallocated in destructor
  //   Data set in ReadParameterFile routines OR by KIM Simulator
  double * cutoff_;
  // lj parameters
  double lj_A_;
  double lj_r_up_min_;
  double lj_r_up_max_;
  double lj_r_down_min_;
  double lj_r_down_max_;
  double lj_cutoff_;
  double ** cutoffSq_2D_;

  // Mutable values that only change when Refresh() executes
  //   Set in Refresh (via SetRefreshMutableValues)
  //
  //
  // KIM API: Model Parameters (can be changed directly by KIM Simulator)
  // none
  //
  // ANNImplementation: values (changed only by Refresh())
  double influenceDistance_;
  int modelWillNotRequestNeighborsOfNoncontributingParticles_;

  // Mutable values that can change with each call to Refresh() and Compute()
  //   Memory may be reallocated on each call
  //
  //
  // ANNImplementation: values that change
  int cachedNumberOfParticles_;

  // descriptor;
  Descriptor * descriptor_;
  NeuralNetwork * network_;

  // Helper methods
  //
  //
  // Related to constructor
  void AllocatePrivateParameterMemory();
  void AllocateParameterMemory();

  static int
  OpenParameterFiles(KIM::ModelDriverCreate * const modelDriverCreate,
                     int const numberParameterFiles,
                     FILE * parameterFilePointers[MAX_PARAMETER_FILES]);
  int ProcessParameterFiles(
      KIM::ModelDriverCreate * const modelDriverCreate,
      int const numberParameterFiles,
      FILE * const parameterFilePointers[MAX_PARAMETER_FILES]);
  void getNextDataLine(FILE * const filePtr,
                       char * const nextLine,
                       int const maxSize,
                       int * endOfFileFlag);
  int getXdouble(char * linePtr, const int N, double * list);
  int getXint(char * linePtr, const int N, int * list);
  void lowerCase(char * linePtr);
  static void
  CloseParameterFiles(int const numberParameterFiles,
                      FILE * const parameterFilePointers[MAX_PARAMETER_FILES]);
  int ConvertUnits(KIM::ModelDriverCreate * const modelDriverCreate,
                   KIM::LengthUnit const requestedLengthUnit,
                   KIM::EnergyUnit const requestedEnergyUnit,
                   KIM::ChargeUnit const requestedChargeUnit,
                   KIM::TemperatureUnit const requestedTemperatureUnit,
                   KIM::TimeUnit const requestedTimeUnit);
  int RegisterKIMModelSettings(
      KIM::ModelDriverCreate * const modelDriverCreate) const;
  int RegisterKIMComputeArgumentsSettings(
      KIM::ModelComputeArgumentsCreate * const modelComputeArgumentsCreate)
      const;
  int RegisterKIMParameters(KIM::ModelDriverCreate * const modelDriverCreate);
  int RegisterKIMFunctions(
      KIM::ModelDriverCreate * const modelDriverCreate) const;

  //
  // Related to Refresh()
  template<class ModelObj>
  int SetRefreshMutableValues(ModelObj * const modelObj);

  //
  // Related to Compute()
  int SetComputeMutableValues(
      KIM::ModelComputeArguments const * const modelComputeArguments,
      bool & isComputeProcess_dEdr,
      bool & isComputeProcess_d2Edr2,
      bool & isComputeEnergy,
      bool & isComputeForces,
      bool & isComputeParticleEnergy,
      bool & isComputeVirial,
      bool & isComputeParticleVirial,
      int const *& particleSpeciesCodes,
      int const *& particleContributing,
      VectorOfSizeDIM const *& coordinates,
      double *& energy,
      VectorOfSizeDIM *& forces,
      double *& particleEnergy,
      VectorOfSizeSix *& virial,
      VectorOfSizeSix *& particleViral);
  int CheckParticleSpeciesCodes(KIM::ModelCompute const * const modelCompute,
                                int const * const particleSpeciesCodes) const;
  int GetComputeIndex(const bool & isComputeProcess_dEdr,
                      const bool & isComputeProcess_d2Edr2,
                      const bool & isComputeEnergy,
                      const bool & isComputeForces,
                      const bool & isComputeParticleEnergy,
                      const bool & isComputeVirial,
                      const bool & isComputeParticleVirial) const;

  // compute functions
  template<bool isComputeProcess_dEdr,
           bool isComputeProcess_d2Edr2,
           bool isComputeEnergy,
           bool isComputeForces,
           bool isComputeParticleEnergy,
           bool isComputeVirial,
           bool isComputeParticleVirial>
  int Compute(KIM::ModelCompute const * const modelCompute,
              KIM::ModelComputeArguments const * const modelComputeArguments,
              const int * const particleSpeciesCodes,
              const int * const particleContributing,
              const VectorOfSizeDIM * const coordinates,
              double * const energy,
              VectorOfSizeDIM * const forces,
              double * const particleEnergy,
              VectorOfSizeSix virial,
              VectorOfSizeSix * const particleVirial) const;

  // LJ functions
  void calc_phi(double const epsilon,
                double const sigma,
                double const cutoff,
                double const r,
                double * const phi) const;
  void calc_phi_dphi(double const epsilon,
                     double const sigma,
                     double const cutoff,
                     double const r,
                     double * const phi,
                     double * const dphi) const;
  void switch_fn(double const x_min,
                 double const x_max,
                 double const x,
                 double * const fn,
                 double * const fn_prime) const;
};

//==============================================================================
//
// Definition of ANNImplementation::Compute functions
//
// NOTE: Here we rely on the compiler optimizations to prune dead code
//       after the template expansions.  This provides high efficiency
//       and easy maintenance.
//
//==============================================================================

#undef KIM_LOGGER_OBJECT_NAME
#define KIM_LOGGER_OBJECT_NAME modelCompute
template<bool isComputeProcess_dEdr,
         bool isComputeProcess_d2Edr2,
         bool isComputeEnergy,
         bool isComputeForces,
         bool isComputeParticleEnergy,
         bool isComputeVirial,
         bool isComputeParticleVirial>
int ANNImplementation::Compute(
    KIM::ModelCompute const * const modelCompute,
    KIM::ModelComputeArguments const * const modelComputeArguments,
    const int * const particleSpeciesCodes,
    const int * const particleContributing,
    const VectorOfSizeDIM * const coordinates,
    double * const energy,
    VectorOfSizeDIM * const forces,
    double * const particleEnergy,
    VectorOfSizeSix virial,
    VectorOfSizeSix * const particleVirial) const
{
  int ier = false;

  if ((isComputeEnergy == false) && (isComputeParticleEnergy == false)
      && (isComputeForces == false) && (isComputeProcess_dEdr == false)
      && (isComputeProcess_d2Edr2 == false) && (isComputeVirial == false)
      && (isComputeParticleVirial == false))
  { return ier; }

  bool need_dE = (isComputeProcess_dEdr == true) || (isComputeForces == true);

  // ANNImplementation: values that does not change
  int const Nparticles = cachedNumberOfParticles_;

  // initialize energy and forces
  if (isComputeEnergy == true) { *energy = 0.0; }

  if (isComputeParticleEnergy == true)
  {
    for (int i = 0; i < Nparticles; ++i) { particleEnergy[i] = 0.0; }
  }

  if (isComputeForces == true)
  {
    for (int i = 0; i < Nparticles; ++i)
    {
      for (int j = 0; j < DIM; ++j) { forces[i][j] = 0.0; }
    }
  }

  if (isComputeVirial == true)
  {
    for (int i = 0; i < 6; ++i) { virial[i] = 0.0; }
  }

  if (isComputeParticleVirial == true)
  {
    for (int i = 0; i < Nparticles; ++i)
    {
      for (int j = 0; j < 6; ++j) { particleVirial[i][j] = 0.0; }
    }
  }

  // Allocate memory for precompute values of sym g4
  size_t n_lambda = descriptor_->g4_distinct_lambda.size();
  size_t n_zeta = descriptor_->g4_distinct_zeta.size();
  size_t n_eta = descriptor_->g4_distinct_eta.size();
  double ** costerm;
  double *** dcosterm_dr;
  double * eterm;
  double ** determ_dr;
  AllocateAndInitialize2DArray<double>(costerm, n_lambda, n_zeta);
  AllocateAndInitialize3DArray<double>(dcosterm_dr, n_lambda, n_zeta, 3);
  AllocateAndInitialize1DArray<double>(eterm, n_eta);
  AllocateAndInitialize2DArray<double>(determ_dr, n_eta, 3);

  // number of descriptors
  const int Ndescriptors = descriptor_->get_num_descriptors();
  const int Ndescriptors_two = descriptor_->get_num_descriptors_two_body();
  const int Ndescriptors_three = descriptor_->get_num_descriptors_three_body();
#ifdef DEBUG
  std::cout << "@Ndescriptors = " << Ndescriptors << std::endl;
  std::cout << "@Ndescriptors_two = " << Ndescriptors_two << std::endl;
  std::cout << "@Ndescriptors_three = " << Ndescriptors_three << std::endl;
#endif

  // index map between 1D two-body, three-body descriptors and global 1D
  // descriptor
  int * map_t_desc_two = new int[Ndescriptors_two];
  int * map_t_desc_three = new int[Ndescriptors_three];
  int t_two = 0;
  int t_three = 0;
  for (size_t p = 0; p < descriptor_->name.size(); p++)
  {
    for (int q = 0; q < descriptor_->num_param_sets[p]; q++)
    {
      if (strcmp(descriptor_->name[p], "g1") == 0
          || strcmp(descriptor_->name[p], "g2") == 0
          || strcmp(descriptor_->name[p], "g3") == 0)
      {
        map_t_desc_two[t_two] = descriptor_->get_global_1D_index(p, q);
        t_two += 1;
      }
      else if (strcmp(descriptor_->name[p], "g4") == 0
               || strcmp(descriptor_->name[p], "g5") == 0)
      {
        map_t_desc_three[t_three] = descriptor_->get_global_1D_index(p, q);
        t_three += 1;
      }
    }
  }

  // allocate memory based on approximate number of neighbors
  // memory will be relallocated if numnei is larger than approx_numnei
  double ** dGCdr_two;
  double *** dGCdr_three;
  int approx_numnei = 100;
  int Npairs_two = approx_numnei;
  int Npairs_three = approx_numnei * (approx_numnei - 1) / 2;
  AllocateAndInitialize2DArray<double>(dGCdr_two, Npairs_two, Ndescriptors_two);
  AllocateAndInitialize3DArray<double>(
      dGCdr_three, Npairs_three, Ndescriptors_three, 3);

  // calculate generalized coordinates
  //
  // Setup loop over contributing particles
  for (int i = 0; i < Nparticles; i++)
  {
    if (!particleContributing[i]) { continue; }

    // get neighbors of atom i
    int numnei = 0;
    int const * n1atom = 0;
    modelComputeArguments->GetNeighborList(0, i, &numnei, &n1atom);
    int const iSpecies = particleSpeciesCodes[i];

    // generalized coords of atom i and its derivatives w.r.t. pair distances
    double * GC;
    AllocateAndInitialize1DArray<double>(GC, Ndescriptors);

    int const Npairs_two = numnei;
    int const Npairs_three = numnei * (numnei - 1) / 2;
    // realloate memory is numnei is larger than approx_numnei
    if (numnei > approx_numnei)
    {
      Deallocate2DArray(dGCdr_two);
      Deallocate3DArray(dGCdr_three);
      AllocateAndInitialize2DArray<double>(
          dGCdr_two, Npairs_two, Ndescriptors_two);
      AllocateAndInitialize3DArray<double>(
          dGCdr_three, Npairs_three, Ndescriptors_three, 3);
      approx_numnei = numnei;
    }

    // Setup loop over neighbors of current particle
    for (int jj = 0; jj < numnei; ++jj)
    {
      // adjust index of particle neighbor
      int const j = n1atom[jj];
      int const jContrib = particleContributing[j];

      int const jSpecies = particleSpeciesCodes[j];

      // cutoff between ij
      double rcutij = sqrt(cutoffSq_2D_[iSpecies][jSpecies]);

      // Compute rij
      double rij[DIM];
      for (int dim = 0; dim < DIM; ++dim)
      { rij[dim] = coordinates[j][dim] - coordinates[i][dim]; }
      double const rijsq = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
      double const rijmag = sqrt(rijsq);

      // lj part

      if (!(jContrib && (j < i)))
      {  // effective half-list
        if (rijmag > lj_cutoff_) { continue; }

        double phi;
        double dphi;
        double dEidr;
        if (need_dE)
        {
          // compute pair potential and its derivative
          const double epsilon = lj_A_ / 4.0;
          const double sigma = 1.;
          calc_phi_dphi(epsilon, sigma, lj_cutoff_, rijmag, &phi, &dphi);

          // switch short range
          double s;
          double ps;
          switch_fn(lj_r_up_min_, lj_r_up_max_, rijmag, &s, &ps);
          double s_up = 1 - s;
          double ps_up = -ps;

          // switch cutoff
          double s_down;
          double ps_down;
          switch_fn(lj_r_down_min_, lj_r_down_max_, rijmag, &s_down, &ps_down);

          dphi = dphi * s_up * s_down + phi * ps_up * s_down
                 + phi * s_up * ps_down;
          phi = phi * s_up * s_down;

          if (jContrib == 1) { dEidr = 2 * dphi; }
          else
          {
            dEidr = dphi;
          }
        }
        else
        {
          const double epsilon = lj_A_ / 4.0;
          const double sigma = 1.;
          calc_phi(epsilon, sigma, lj_cutoff_, rijmag, &phi);
          double s;
          double ps;
          switch_fn(lj_r_up_min_, lj_r_up_max_, rijmag, &s, &ps);
          double s_up = 1 - s;

          // switch cutoff
          double s_down;
          double ps_down;
          switch_fn(lj_r_down_min_, lj_r_down_max_, rijmag, &s_down, &ps_down);
          phi = phi * s_up * s_down;
        }

        // particle energy
        if (isComputeParticleEnergy)
        {
          particleEnergy[i] += phi;
          if (jContrib == 1) { particleEnergy[j] += phi; }
        }

        // energy
        if (isComputeEnergy)
        {
          if (jContrib == 1) { *energy += 2 * phi; }
          else
          {
            *energy += phi;
          }
        }

        // forces
        if (isComputeForces)
        {
          for (int k = 0; k < DIM; ++k)
          {
            forces[i][k] += dEidr * rij[k] / rijmag;
            forces[j][k] -= dEidr * rij[k] / rijmag;
          }
        }

        //  virial
        if (isComputeVirial == true)
        { ProcessVirialTerm(dEidr, rijmag, rij, i, j, virial); }

        //  particleVirial
        if (isComputeParticleVirial == true)
        {
          ProcessParticleVirialTerm(dEidr, rijmag, rij, i, j, particleVirial);
        }

        // process_dEdr
        if (isComputeProcess_dEdr == true)
        {
          ier = modelComputeArguments->ProcessDEDrTerm(
              dEidr, rijmag, rij, i, j);
          if (ier)
          {
            LOG_ERROR("ProcessDEdr");
            return ier;
          }
        }
      }  // rij < cutoff

      // NN part

      // if particles i and j not interact
      if (rijmag > rcutij) { continue; }

      // pre-compute two-body cut function
      double fcij = descriptor_->cutoff(rijmag, rcutij);
      double dfcij = descriptor_->d_cutoff(rijmag, rcutij);

      int s_two = jj;  // row index of dGCdr_two
      int t_two = 0;  // column index of dGCdr_two
      for (size_t p = 0; p < descriptor_->name.size(); p++)
      {
        if (strcmp(descriptor_->name[p], "g1") != 0
            && strcmp(descriptor_->name[p], "g2") != 0
            && strcmp(descriptor_->name[p], "g3") != 0)
        { continue; }

        for (int q = 0; q < descriptor_->num_param_sets[p]; q++)
        {
          double gc;
          double dgcdr_two;

          //          if (strcmp(descriptor_->name[p], "g1") == 0) {
          //            if (need_dE) {
          //              descriptor_->sym_d_g1(rijmag, rcutij, gc, dgcdr_two);
          //            } else {
          //              descriptor_->sym_g1(rijmag, rcutij, gc);
          //            }
          //          }
          //          else if (strcmp(descriptor_->name[p], "g2") == 0) {
          double eta = descriptor_->params[p][q][0];
          double Rs = descriptor_->params[p][q][1];

          //          if (need_dE) {
          descriptor_->sym_d_g2(
              eta, Rs, rijmag, rcutij, fcij, dfcij, gc, dgcdr_two);
          //          } else {
          //            descriptor_->sym_g2(eta, Rs, rijmag, rcutij, gc);
          //          }
          //          }
          //          else if (strcmp(descriptor_->name[p], "g3") == 0) {
          //            double kappa = descriptor_->params[p][q][0];
          //            if (need_dE) {
          //              descriptor_->sym_d_g3(kappa, rijmag, rcutij, gc,
          //              dgcdr_two);
          //            } else {
          //              descriptor_->sym_g3(kappa, rijmag, rcutij, gc);
          //            }
          //          }
          //

          int desc_idx = descriptor_->get_global_1D_index(p, q);
          GC[desc_idx] += gc;
          //          if (need_dE) {
          dGCdr_two[s_two][t_two] = dgcdr_two;
          t_two += 1;
          //          }
        }  // loop over same descriptor but different parameter set
      }  // loop over descriptors

      // three-body descriptors
      //      if (descriptor_->has_three_body == false) continue;

      for (int kk = jj + 1; kk < numnei; ++kk)
      {
        // adjust index of particle neighbor
        int const k = n1atom[kk];
        int const kSpecies = particleSpeciesCodes[k];

        // cutoff between ik and jk
        double const rcutik = sqrt(cutoffSq_2D_[iSpecies][kSpecies]);
        double const rcutjk = sqrt(cutoffSq_2D_[jSpecies][kSpecies]);

        // Compute rik, rjk and their squares
        double rik[DIM];
        double rjk[DIM];
        for (int dim = 0; dim < DIM; ++dim)
        {
          rik[dim] = coordinates[k][dim] - coordinates[i][dim];
          rjk[dim] = coordinates[k][dim] - coordinates[j][dim];
        }
        double const riksq
            = rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2];
        double const rjksq
            = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];
        double const rikmag = sqrt(riksq);
        double const rjkmag = sqrt(rjksq);

        double const rvec[3] = {rijmag, rikmag, rjkmag};
        double const rcutvec[3] = {rcutij, rcutik, rcutjk};

        if (rikmag > rcutik)
        {
          continue;  // three-dody not interacting
        }
        // TODO only for g4, should delete this if we have g5
        if (rjkmag > rcutjk)
        {
          continue;  // only for g4, not for g4
        }
        // cutoff term, i.e. the product of fc(rij), fc(rik), and fc(rjk)
        double fcik = cut_cos(rikmag, rcutik);
        double fcjk = cut_cos(rjkmag, rcutjk);
        double dfcik = d_cut_cos(rikmag, rcutik);
        double dfcjk = d_cut_cos(rjkmag, rcutjk);
        double fcprod = fcij * fcik * fcjk;
        double dfcprod_dr[3];  // dfcprod/drij, dfcprod/drik, dfcprod/drjk
        dfcprod_dr[0] = dfcij * fcik * fcjk;
        dfcprod_dr[1] = dfcik * fcij * fcjk;
        dfcprod_dr[2] = dfcjk * fcij * fcik;

        int s_three = (numnei - 1 + numnei - jj) * jj / 2
                      + (kk - jj - 1);  // row index of dGCdr_three
#ifdef DEBUG
        std::cout << "@numnei=" << numnei << std::endl;
        std::cout << "@jj=" << jj << std::endl;
        std::cout << "@kk=" << kk << std::endl;
        std::cout << "@s_three=" << s_three << std::endl;
#endif

        int t_three = 0;  // column index of dGCdr_three

        for (size_t p = 0; p < descriptor_->name.size(); p++)
        {
          if (strcmp(descriptor_->name[p], "g4") != 0
              && strcmp(descriptor_->name[p], "g5") != 0)
          { continue; }

          // precompute recurring values in cosine terms and exponential terms
          descriptor_->precompute_g4(rijmag,
                                     rikmag,
                                     rjkmag,
                                     rijsq,
                                     riksq,
                                     rjksq,
                                     n_lambda,
                                     n_zeta,
                                     n_eta,
                                     costerm,
                                     dcosterm_dr,
                                     eterm,
                                     determ_dr);

          for (int q = 0; q < descriptor_->num_param_sets[p]; q++)
          {
            double gc;
            double dgcdr_three[3];

            //            if (strcmp(descriptor_->name[p], "g4") == 0) {

            //              if (need_dE) {

            // get values from precomputed
            int izeta = descriptor_->g4_lookup_zeta[q];
            int ilam = descriptor_->g4_lookup_lambda[q];
            int ieta = descriptor_->g4_lookup_eta[q];

            double ct = costerm[ilam][izeta];
            double dct[3];
            dct[0] = dcosterm_dr[ilam][izeta][0];
            dct[1] = dcosterm_dr[ilam][izeta][1];
            dct[2] = dcosterm_dr[ilam][izeta][2];

            double et = eterm[ieta];
            double det[3];
            det[0] = determ_dr[ieta][0];
            det[1] = determ_dr[ieta][1];
            det[2] = determ_dr[ieta][2];

            descriptor_->sym_d_g4_2(rvec,
                                    rcutvec,
                                    fcprod,
                                    dfcprod_dr,
                                    ct,
                                    dct,
                                    et,
                                    det,
                                    gc,
                                    dgcdr_three);

            //              } else {
            //                descriptor_->sym_g4(zeta, lambda, eta, rvec,
            //                rcutvec, gc);
            //              }
            //            }
            //            else if (strcmp(descriptor_->name[p], "g5") == 0) {
            //              double zeta = descriptor_->params[p][q][0];
            //              double lambda = descriptor_->params[p][q][1];
            //              double eta = descriptor_->params[p][q][2];
            //              if (need_dE) {
            //                descriptor_->sym_d_g5(zeta, lambda, eta, rvec,
            //                rcutvec, gc, dgcdr_three);
            //              } else {
            //                descriptor_->sym_g5(zeta, lambda, eta, rvec,
            //                rcutvec, gc);
            //              }
            //            }
            //

            int desc_idx = descriptor_->get_global_1D_index(p, q);
            GC[desc_idx] += gc;
            //            if (need_dE) {
            dGCdr_three[s_three][t_three][0] = dgcdr_three[0];
            dGCdr_three[s_three][t_three][1] = dgcdr_three[1];
            dGCdr_three[s_three][t_three][2] = dgcdr_three[2];
            t_three += 1;
            //            }
          }  // loop over same descriptor but different parameter set
        }  // loop over descriptors
      }  // loop over kk (three body neighbors)
    }  // loop over jj

    // centering and normalization
    if (descriptor_->center_and_normalize)
    {
      for (int t = 0; t < Ndescriptors; t++)
      {
        GC[t] = (GC[t] - descriptor_->features_mean[t])
                / descriptor_->features_std[t];
      }

      if (need_dE)
      {
        for (int s = 0; s < Npairs_two; s++)
        {
          for (int t = 0; t < Ndescriptors_two; t++)
          {
            int desc_idx = map_t_desc_two[t];
            dGCdr_two[s][t] /= descriptor_->features_std[desc_idx];
          }
        }
        for (int s = 0; s < Npairs_three; s++)
        {
          for (int t = 0; t < Ndescriptors_three; t++)
          {
            int desc_idx = map_t_desc_three[t];
            dGCdr_three[s][t][0] /= descriptor_->features_std[desc_idx];
            dGCdr_three[s][t][1] /= descriptor_->features_std[desc_idx];
            dGCdr_three[s][t][2] /= descriptor_->features_std[desc_idx];
          }
        }
      }
    }

#ifdef DEBUG
    // generalized coords
    std::cout << "\n# Debug descriptor values after normalization" << std::endl;
    std::cout << "# atom id    descriptor values ..." << std::endl;
    std::cout << ii << "    ";
    for (int j = 0; j < Ndescriptors; j++) { printf("%.15f ", GC[j]); }
    std::cout << std::endl;
#endif

    // NN feedforward
    network_->forward(GC, 1, Ndescriptors);

    // NN backpropagation to compute derivative of energy w.r.t generalized
    // coords
    double * dEdGC;
    if (need_dE)
    {
      network_->backward();
      dEdGC = network_->get_grad_input();
    }

    double Ei = 0.;
    if (isComputeEnergy == true || isComputeParticleEnergy == true)
    { Ei = network_->get_sum_output(); }

    // Contribution to energy
    if (isComputeEnergy == true) { *energy += Ei; }

    // Contribution to particle energy
    if (isComputeParticleEnergy == true) { particleEnergy[i] += Ei; }

    // Contribution to forces and virial
    if (need_dE)
    {
      // neighboring atoms of i
      for (int jj = 0; jj < numnei; ++jj)
      {
        int const j = n1atom[jj];
        int const jSpecies = particleSpeciesCodes[j];
        double rcutij = sqrt(cutoffSq_2D_[iSpecies][jSpecies]);

        // Compute rij
        double rij[DIM];
        for (int dim = 0; dim < DIM; ++dim)
        { rij[dim] = coordinates[j][dim] - coordinates[i][dim]; }
        double const rijsq
            = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
        double const rijmag = sqrt(rijsq);

        // if particles i and j not interact
        if (rijmag > rcutij) { continue; }

        // two-body descriptors
        int s_two = jj;
        double dEdr_two = 0;
        for (int t = 0; t < Ndescriptors_two; t++)
        {
          int desc_idx = map_t_desc_two[t];
          dEdr_two += dGCdr_two[s_two][t] * dEdGC[desc_idx];
        }

        // forces
        if (isComputeForces)
        {
          for (int dim = 0; dim < DIM; ++dim)
          {
            double pair = dEdr_two * rij[dim] / rijmag;
            forces[i][dim] += pair;  // for i atom
            forces[j][dim] -= pair;  // for neighboring atoms of i
          }
        }

        //  virial
        if (isComputeVirial == true)
        { ProcessVirialTerm(dEdr_two, rijmag, rij, i, j, virial); }

        //  particleVirial
        if (isComputeParticleVirial == true)
        {
          ProcessParticleVirialTerm(
              dEdr_two, rijmag, rij, i, j, particleVirial);
        }

        // process_dEdr
        if (isComputeProcess_dEdr == true)
        {
          ier = modelComputeArguments->ProcessDEDrTerm(
              dEdr_two, rijmag, rij, i, j);
          if (ier)
          {
            LOG_ERROR("ProcessDEdr");
            return ier;
          }
        }

        // three-body descriptors
        for (int kk = jj + 1; kk < numnei; ++kk)
        {
          int const k = n1atom[kk];
          int const kSpecies = particleSpeciesCodes[k];

          // cutoff between ik and jk
          double const rcutik = sqrt(cutoffSq_2D_[iSpecies][kSpecies]);
          double const rcutjk = sqrt(cutoffSq_2D_[jSpecies][kSpecies]);

          // Compute rik, rjk and their squares
          double rik[DIM];
          double rjk[DIM];
          for (int dim = 0; dim < DIM; ++dim)
          {
            rik[dim] = coordinates[k][dim] - coordinates[i][dim];
            rjk[dim] = coordinates[k][dim] - coordinates[j][dim];
          }
          double const riksq
              = rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2];
          double const rjksq
              = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];
          double const rikmag = sqrt(riksq);
          double const rjkmag = sqrt(rjksq);

          if (rikmag > rcutik)
          {
            continue;  // three-dody not interacting
          }
          // TODO only for g4, should delete this if we have g5
          if (rjkmag > rcutjk)
          {
            continue;  // only for g4, not for g4
          }
          int s_three = (numnei - 1 + numnei - jj) * jj / 2
                        + (kk - jj - 1);  // row index of dGCdr_three
          double dEdr_three[3] = {0, 0, 0};
          for (int t = 0; t < Ndescriptors_three; t++)
          {
            int desc_idx = map_t_desc_three[t];
            dEdr_three[0]
                += dGCdr_three[s_three][t][0] * dEdGC[desc_idx];  // dEdrij
            dEdr_three[1]
                += dGCdr_three[s_three][t][1] * dEdGC[desc_idx];  // dEdrik
            dEdr_three[2]
                += dGCdr_three[s_three][t][2] * dEdGC[desc_idx];  // dEdrjk
          }

          // forces
          if (isComputeForces)
          {
            for (int dim = 0; dim < DIM; ++dim)
            {
              double pair_ij = dEdr_three[0] * rij[dim] / rijmag;
              double pair_ik = dEdr_three[1] * rik[dim] / rikmag;
              double pair_jk = dEdr_three[2] * rjk[dim] / rjkmag;
              forces[i][dim] += pair_ij + pair_ik;  // for i atom
              forces[j][dim]
                  += -pair_ij + pair_jk;  // for neighboring atoms of i
              forces[k][dim]
                  += -pair_ik - pair_jk;  // for neighboring atoms of i
            }
          }

          // virial
          if (isComputeVirial == true)
          {
            ProcessVirialTerm(dEdr_three[0], rijmag, rij, i, j, virial);
            ProcessVirialTerm(dEdr_three[1], rikmag, rik, i, k, virial);
            ProcessVirialTerm(dEdr_three[2], rjkmag, rjk, j, k, virial);
          }

          // particleVirial
          if (isComputeParticleVirial == true)
          {
            ProcessParticleVirialTerm(
                dEdr_three[0], rijmag, rij, i, j, particleVirial);
            ProcessParticleVirialTerm(
                dEdr_three[1], rikmag, rik, i, k, particleVirial);
            ProcessParticleVirialTerm(
                dEdr_three[2], rjkmag, rjk, j, k, particleVirial);
          }

          // process_dEdr
          if (isComputeProcess_dEdr == true)
          {
            ier = modelComputeArguments->ProcessDEDrTerm(
                      dEdr_three[0], rijmag, rij, i, j)
                  || modelComputeArguments->ProcessDEDrTerm(
                      dEdr_three[1], rikmag, rik, i, k)
                  || modelComputeArguments->ProcessDEDrTerm(
                      dEdr_three[2], rjkmag, rjk, j, k);
            if (ier)
            {
              LOG_ERROR("ProcessDEdr");
              return ier;
            }
          }
        }  // loop over kk
      }  // loop over jj
    }  // need_dE

    Deallocate1DArray(GC);
  }  // loop over ii, i.e. contributing particles

  Deallocate2DArray(costerm);
  Deallocate3DArray(dcosterm_dr);
  Deallocate1DArray(eterm);
  Deallocate2DArray(determ_dr);
  delete[] map_t_desc_two;
  delete[] map_t_desc_three;

  Deallocate2DArray(dGCdr_two);
  Deallocate3DArray(dGCdr_three);

  // everything is good
  ier = false;
  return ier;
}

#endif  // ANN_IMPLEMENTATION_HPP_
