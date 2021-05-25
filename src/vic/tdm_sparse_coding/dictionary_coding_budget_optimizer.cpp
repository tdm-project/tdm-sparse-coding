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
#include <vic/tdm_sparse_coding/dictionary_coding_budget_optimizer.hpp>
#include <vic/tdm_sparse_coding/matching_pursuit.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {
    
    dictionary_coding_budget_optimizer* dictionary_coding_budget_optimizer::clone() const {
      return new dictionary_coding_budget_optimizer(*this);
    }

    void dictionary_coding_budget_optimizer::optimize_in(std::vector<std::size_t>&  gamma_offset,
							 std::vector<std::size_t>&  gamma_sz,
							 std::vector<float>&        gamma_val,
							 std::vector<std::size_t>&  gamma_idx,
							 const std::vector<Eigen::VectorXf>& Y) {
      const std::size_t Y_count = Y.size();
      const std::size_t S = non_zeros_per_signal_;
      
      gamma_offset.clear();
      gamma_sz.clear();
      gamma_val.clear();
      gamma_idx.clear();

      coder_->precompute_coding_data();
      
      const std::size_t BATCH_SIZE=8192;
      std::vector<std::size_t>  batch_gamma_sz(BATCH_SIZE);
      std::vector<float>        batch_gamma_val(BATCH_SIZE*S);
      std::vector<std::size_t>  batch_gamma_idx(BATCH_SIZE*S);
      for (std::size_t i_bgn=0; i_bgn<Y_count; i_bgn+=BATCH_SIZE) {
	std::size_t i_end = std::min(Y_count, i_bgn+BATCH_SIZE);
	
#pragma omp parallel for
	for (std::size_t i=i_bgn; i<i_end; ++i) {
	  const Eigen::VectorXf& y_i = Y[i];

	  std::size_t S_out = 0;
	  coder_->encode_in(&(batch_gamma_idx[(i-i_bgn)*S]),
			    &(batch_gamma_val[(i-i_bgn)*S]),
			    y_i,
			    S,
			    &S_out);
	  // Default is coder's setting 
	  batch_gamma_sz[i-i_bgn] = S_out; // S
	}
	
	// Preallocate output data after first batch
	if (i_bgn==0) {
	  gamma_sz.reserve(Y_count);
	  gamma_offset.reserve(Y_count);
	  std::size_t gamma_count = Y_count*S;
	  // Allocate
	  gamma_val.reserve(gamma_count);
	  gamma_idx.reserve(gamma_count);
	}
	// Merge batch into output
	for (std::size_t i=i_bgn; i<i_end; ++i) {
	  const std::size_t k_i = batch_gamma_sz[i-i_bgn];
	  gamma_sz.push_back(k_i);
	  gamma_offset.push_back(gamma_val.size());
	  for (std::size_t k=0; k<k_i; ++k) {
	    gamma_val.push_back(batch_gamma_val[(i-i_bgn)*S+k]);
	    gamma_idx.push_back(batch_gamma_idx[(i-i_bgn)*S+k]);
	  }
	}
      } // For each batch
    }

    void dictionary_coding_budget_optimizer::decode_in(Eigen::VectorXf& y_tilde,
						       const std::size_t *gamma_idx,
						       const float *gamma_val,
						       std::size_t S) const {
      if (S == std::size_t(-1)) S = non_zeros_per_signal_;

      coder_->decode_in(y_tilde, gamma_idx, gamma_val, S);
    }
	  
    void dictionary_coding_budget_optimizer::approximate_in(std::vector<Eigen::VectorXf>& Y_tilde,
							    const std::vector<Eigen::VectorXf>& Y,
							    std::size_t* gamma_count) {
      std::vector<std::size_t> gamma_offset;
      std::vector<std::size_t> gamma_sz;
      std::vector<float>       gamma_val;
      std::vector<std::size_t> gamma_idx;

      optimize_in(gamma_offset, gamma_sz, gamma_val, gamma_idx, Y);
      if (gamma_count) *gamma_count = gamma_idx.size();
			 
      const std::size_t R=Y.size();
      Y_tilde.resize(R, Y[0]);
#pragma omp parallel for
      for (std::size_t r=0; r<R; ++r) {
	std::size_t i_r=gamma_offset[r];
	std::size_t S_r = gamma_sz[r];
	decode_in(Y_tilde[r],
		  &(gamma_idx[i_r]),
		  &(gamma_val[i_r]),
		  S_r);
      }
    }

  } // namespace tdm_sparse_coding
} // namespace vic
