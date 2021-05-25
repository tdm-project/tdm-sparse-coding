// +++HDR+++
// ======================================================================
// 
//   This file is part of the TDM SPARSE CODING software library.
// 
//   Main authors: Enrico Gobbetti & Fabio Marton
// 
//   Copyright (C) 2021 by CRS4, Cagliari, Italy
// 
//   For more information, visit the CRS4 Visual Computing 
//   web pages at http://www.crs4.it/vic/ and the TDM project web
//   pages at  http://www.tdm-project.it
// 
//   The software is distributed under the CC BY-NC-ND 4.0 licence.
//   https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
//  
//   For use in commercial projects, kindly contact gobbetti@crs4.it.
//   If you use this software for research or in a publication, kindly
//   cite the original references. 
// 
//   CRS4 reserves all rights not expressly granted herein.
//   
//   This file is provided AS IS with NO WARRANTY OF ANY KIND, 
//   INCLUDING THE WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS 
//   FOR A PARTICULAR PURPOSE.
// 
// ======================================================================
// ---HDR---
#ifndef VIC_TDM_SPARSE_CODING_OMP_HPP
#define VIC_TDM_SPARSE_CODING_OMP_HPP

#include <sl/config.hpp>
#include <Eigen/Core>
#include <vector>

namespace vic {
 
  /*
    ----------------------------------------------------------------------
    Helpers
    ----------------------------------------------------------------------
  */

  namespace tdm_sparse_coding {   
    extern void least_squares_in(float *gamma_val,
				 const std::size_t *gamma_idx,
				 const Eigen::MatrixXf& D,
				 const Eigen::VectorXf& x,
				 std::size_t K);
  }
  
  /*
    ----------------------------------------------------------------------
    Batch Orthogonal matching pursuit
    
    References:
  
    [1] M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm
    for Designing of Overcomplete Dictionaries for Sparse
    Representation", the IEEE Trans. On Signal Processing, Vol. 54, no.
    11, pp. 4311-4322, November 2006.

    [2] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation
    of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit",
    Technical Report - CS, Technion, April 2008.
    ----------------------------------------------------------------------
  */
  
  namespace tdm_sparse_coding {
    // Single vector
    extern void omp_batch_in(float       *gamma_val,
			     std::size_t *gamma_idx,
			     const Eigen::VectorXf& Dtx, // = trans(D)*x
			     const Eigen::MatrixXf& DtD,
			     std::size_t K,
			     float       xtx=0.0f,            // = trans(x)*x
			     float tol=0.0f,
			     std::size_t* out_K =0, // Achieved sparsity
			     float* out_err2 = 0,  // Achieved error squared
			     float* out_err2_k = 0); // size K+1, on output is the sq error for 0..out_K

    // Group
    extern void omp_batch_in(std::vector<std::size_t>&  gamma_offset,
			     std::vector<std::size_t>&  gamma_sz,
			     std::vector<float>&        gamma_val,
			     std::vector<std::size_t>&  gamma_idx,
			     const Eigen::MatrixXf& D,
			     const Eigen::MatrixXf& DtD,
			     const std::vector<Eigen::VectorXf>& X,
			     std::size_t K,
			     float tol = 0.0f);
  }
  
  /*
    ----------------------------------------------------------------------
    Order Recursive Orthogonal matching pursuit
    
    References:
    
    [1] Natarajan, B. K. Sparse approximate solutions to linear system.
    SIAM Journal on Computing 24, 2 (Apr. 1995), 227–234.
    ----------------------------------------------------------------------
  */
  namespace tdm_sparse_coding {

    extern void ormp_in(float *gamma_val,
			std::size_t *gamma_idx,
			const Eigen::MatrixXf& D,
			const Eigen::VectorXf& x,
			std::size_t K);

    extern void ormp_in(std::vector<float>&        gamma_val,
			std::vector<std::size_t>&  gamma_idx,
			const Eigen::MatrixXf& D,
			const Eigen::VectorXf& x,
			std::size_t K);
    
    extern void ormp_in(Eigen::VectorXf& gamma,
		       const Eigen::MatrixXf& D,
		       const Eigen::VectorXf& x,
		       std::size_t K);

  } // namespace tdm_sparse_coding

    /*
    ----------------------------------------------------------------------
    Orthogonal matching pursuit with replacement (used for refinement only)
    
    References:
    
    [1] Jain, Prateek, Ambuj Tewari, and Inderjit S. Dhillon.
    "Orthogonal matching pursuit with replacement." Advances in Neural
    Information Processing Systems. 2011.
    ----------------------------------------------------------------------
  */
  namespace tdm_sparse_coding {

    extern void ompr_refine_in(float *gamma_val,
			       std::size_t *gamma_idx,
			       std::size_t K,
			       const Eigen::MatrixXf& D,
			       const Eigen::VectorXf& x,
			       const Eigen::VectorXf& Dtx, // = trans(D)*x
			       const Eigen::MatrixXf& DtD);

    extern void ompr_refine_batch_in(std::vector<std::size_t>&  gamma_offset,
				     std::vector<std::size_t>&  gamma_sz,
				     std::vector<float>&        gamma_val,
				     std::vector<std::size_t>&  gamma_idx,
				     const Eigen::MatrixXf& D,
				     const Eigen::MatrixXf& DtD,
				     const std::vector<Eigen::VectorXf>& X);

  } // namespace tdm_sparse_coding

    /*
    ----------------------------------------------------------------------
    Improved matching pursuit (mp tuned for highly coherent dicts)
    
    References:
    
    [1] Sahnoun, Souleymen, Pierre Comon, and Alex Pereira da Silva.
    A Greedy Sparse Method Suitable for Spectral-Line Estimation. Diss. GIPSA-lab,
    2016.
    ----------------------------------------------------------------------
  */
  namespace tdm_sparse_coding {

    extern void imp_refine_in(float *gamma_val,
			      std::size_t *gamma_idx,
			      std::size_t K,
			      const Eigen::MatrixXf& D,
			      const Eigen::VectorXf& x,
			      const Eigen::VectorXf& Dtx, // = trans(D)*x
			      const Eigen::MatrixXf& DtD);

    extern void imp_refine_batch_in(std::vector<std::size_t>&  gamma_offset,
				    std::vector<std::size_t>&  gamma_sz,
				    std::vector<float>&        gamma_val,
				    std::vector<std::size_t>&  gamma_idx,
				    const Eigen::MatrixXf& D,
				    const Eigen::MatrixXf& DtD,
				    const std::vector<Eigen::VectorXf>& X);

  } // namespace tdm_sparse_coding

} // namespace vic

#endif
  
