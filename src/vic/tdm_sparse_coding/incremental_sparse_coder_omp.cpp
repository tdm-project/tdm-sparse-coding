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
#include <vic/tdm_sparse_coding/incremental_sparse_coder_omp.hpp>

namespace vic {

  namespace tdm_sparse_coding {

    void incremental_sparse_coder_omp::clear() {
      // Output
      super_t::clear();

      // State
      clear_state();
    }

    void incremental_sparse_coder_omp::clear_state() {
      state_deltaprev_ = 0.0;
      state_alpha_.resize(0);
      state_Lchol_.resize(0,0);
    }
    
    void incremental_sparse_coder_omp::init(const Eigen::MatrixXf* DtD_ptr,
						       const Eigen::VectorXf& Dty,
						       float yty) {
      assert(DtD_ptr);

      // Reset
      clear();      
      DtD_ptr_ = DtD_ptr;
      Dty_ = Dty;

      // Compute solution for zero coefficients
      err2_.push_back(yty); 

      if (err2_[0]==0.0) {
	// No need to compute - leave state cleared
      } else {
	// Preallocate storage and init computation
	// Init Cholesky decomposition
	
	state_deltaprev_ = 0.0;
	state_Lchol_ = Eigen::MatrixXf::Zero(8,8); 
	state_Lchol_(0,0) = 1.0f;
	state_alpha_ = Dty_;
      }

      // postconditions
      assert(err2_.size() == 1);
      assert(gamma_offset_.size() == 0);
      assert(gamma_val_.size() == 0);
      assert(gamma_idx_.size() == 0);
    }

    void incremental_sparse_coder_omp::null_step() {
      // Same error as previous solution
      err2_.push_back(err2_.back());

      // Same gamma coeffients as previous solution..
      std::size_t k_prev = gamma_offset_.size(); 
      std::size_t offset_k  = gamma_val_.size();
      
      if (k_prev>0) {
	std::size_t offset_k_prev = gamma_offset_.back();
	for (std::size_t k=0; k<k_prev; ++k) {
	  gamma_val_.push_back(gamma_val_[offset_k_prev+k]);
	  gamma_idx_.push_back(gamma_idx_[offset_k_prev+k]);
	}
      }
      
      // ... plus an extra zero.
      gamma_val_.push_back(0.0f);
      gamma_idx_.push_back(0);

      // Increase K
      gamma_offset_.push_back(offset_k);

      // postconditions
      assert(gamma_offset_.size() == k_prev+1);
      assert(gamma_offset_.size()+1 == err2_.size());
      assert(gamma_val_.size() == gamma_offset_.back()+gamma_offset_.size());
      assert(gamma_idx_.size() == gamma_offset_.back()+gamma_offset_.size());
    }

    inline std::size_t iabsdist(std::size_t a, std::size_t b) {
      if (a>b) return a-b; else return b-a;
    }

    inline std::size_t iabsdist(const std::vector<std::size_t>& old_idx,
			 std::size_t new_idx) {
      const std::size_t N = old_idx.size();
      if (N==0) return new_idx;
      std::size_t result = iabsdist(new_idx, old_idx[0]);
      for (std::size_t i=1; i<N; ++i) {
	std::size_t d_i = iabsdist(new_idx, old_idx[i]);
	if (d_i<result) result=d_i;
      }
      return result;
    }
		    
    void incremental_sparse_coder_omp::step() {
      // Compute next solution starting from current one
      const std::size_t Kmax = std::size_t(Dty_.rows());
      const std::size_t k_prev=gamma_offset_.size();

      // Prepare trivial extension with previous coeffs extended with a zero
      std::size_t offset_k  = gamma_val_.size();
      null_step();

      if (state_alpha_.size()==0) {
	// If previous step converged or aborted nothing to do
      } else if (k_prev+1>=Kmax) {
	// Too many steps...
	clear_state();
      } else {
	// Compute last idx of current solutions and all alphas
	

	//////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////
	/////// FIXME FIXME FIXME - TEST LIMIT BITS 
	/////////////////////////////////////////////////////////////////////

	// Find most aligned basis vec
	std::size_t k_star = 0;
	float       a_star = sl::abs(state_alpha_[0]);
	for (std::size_t k1=1; k1<Kmax; ++k1) {
	  const float a = sl::abs(state_alpha_[k1]);
	  if (a>a_star) {
	    a_star = a; k_star = k1;
	  }
	}
	
	if (a_star<1.0e-6f) {
	  // Almost zero -- this step and all future ones will be null
	  clear_state();
	} else if (k_prev==0) {
	  // Compute first approximation
	  // First value -- just dot product

	  assert(gamma_idx_.size() == 1);
	  assert(gamma_val_.size() == 1);
		 
	  gamma_idx_[0] = k_star;
	  gamma_val_[0] = Dty_[k_star];

	  // Update alpha = D r
	  state_alpha_ = Dty_ - (*DtD_ptr_).col(gamma_idx_[0]) * gamma_val_[0];

	  // Compute first error
	  double delta = double(gamma_val_[0]) * (double(Dty_[gamma_idx_[0]]) - double(state_alpha_[gamma_idx_[0]]));
	  err2_[1] = std::max(0.0, err2_[1] - delta);
	  state_deltaprev_=delta;
	} else {
	  // Other values -- must recompute least squares solution using all basis vecs
	    
	  // Incremental Cholesky decomposition: compute next row of Lchol
	  Eigen::VectorXf DtD_Ik(k_prev);
	  for (std::size_t k1=0; k1<k_prev; ++k1) {
	    DtD_Ik(k1) = (*DtD_ptr_)(gamma_idx_[offset_k+k1],k_star);
	  }
	  Eigen::VectorXf w = state_Lchol_.topLeftCorner(k_prev, k_prev).triangularView<Eigen::Lower>().solve(DtD_Ik);
	  const float one_minus_w2 = 1.0f-w.squaredNorm();
	  if (one_minus_w2 <= 1e-14f) {
	    clear_state();
	    // Abort: will keep trivial extension
	  } else {
	    // Grow Choleski decomposition if needed
	    if (state_Lchol_.rows()<int(k_prev+1)) {
	      state_Lchol_.conservativeResize(state_Lchol_.rows()+8,state_Lchol_.cols()+8);
	    }
	    state_Lchol_.block(k_prev,0,1,k_prev) = w.transpose();
	    state_Lchol_(k_prev,k_prev) = std::sqrt(one_minus_w2);

	    // Update idx
	    gamma_idx_[offset_k+k_prev] = k_star;
	    
	    // Compute gamma
	    Eigen::VectorXf Dty_I(k_prev+1);
	    for (std::size_t k1=0; k1<=k_prev; ++k1) {
	      Dty_I(k1) = Dty_(gamma_idx_[offset_k+k1]);
	    }
	    state_Lchol_.topLeftCorner(k_prev+1, k_prev+1).triangularView<Eigen::Lower>().solveInPlace(Dty_I);
	    state_Lchol_.topLeftCorner(k_prev+1, k_prev+1).transpose().triangularView<Eigen::Upper>().solveInPlace(Dty_I);

	    // Push next selected vector and update projection
	    for (std::size_t k1=0; k1<=k_prev; ++k1) {
	      gamma_val_[offset_k+k1] = Dty_I(k1);
	    }
	         
	    // Compute residual similarity with basis vecs
	    // r = x-Dg
	    // similarity = max trans(d_k) * r
	    // trans(D)r = trans(D)*x - trans(D)*D.g = Dty - DtD * g  
	    state_alpha_ = Dty_;
	    for (std::size_t k1=0; k1<=k_prev; ++k1) {
	      state_alpha_ -= (*DtD_ptr_).col(gamma_idx_[offset_k+k1]) * gamma_val_[offset_k+k1];
	    }

	    // From Rubinstein et al, "Efficient implementation of the K-SVD
	    // algorithm using batch orthogonal matching pursuit":
	    //   beta_I = DtD_I * gamma_I
	    //   alpha = Dty - beta_I
	    //   delta = gamma_I^T * beta_I
	    //   err2 = err2 - delta + delta_prev
	    //   delta_prev = delta
	    // My version without vector temporaries:
	    //    alpha = Dty - DtD_I * gamma_I
	    //    delta = gamma_I^T * (Dty - alpha) 
	    //    err2 = err2 - delta + delta_prev
	    //    delta_prev = delta
	  
	    double delta = 0.0;
	    for (std::size_t k1=0; k1<=k_prev; ++k1) {
	      delta += double(gamma_val_[offset_k+k1]) * (double(Dty_[gamma_idx_[offset_k+k1]]) - double(state_alpha_[gamma_idx_[offset_k+k1]]));
	    }
	    err2_.back() = std::max(0.0, err2_.back() - delta + state_deltaprev_);
	    state_deltaprev_=delta;
	  } // if not aborted
	} // if not first step
      } // if not previously aborted

      // postconditions
      assert(gamma_offset_.size() == k_prev+1);
      assert(gamma_offset_.size()+1 == err2_.size());
      assert(gamma_val_.size() == gamma_offset_.back()+gamma_offset_.size());
      assert(gamma_idx_.size() == gamma_offset_.back()+gamma_offset_.size());
    } 

  }
}
