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
#include <vic/tdm_sparse_coding/dictionary_coding_budget_optimizer_greedy_grow.hpp>
#include <vic/tdm_sparse_coding/matching_pursuit.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {
    
    dictionary_coding_budget_optimizer_greedy_grow* dictionary_coding_budget_optimizer_greedy_grow::clone() const {
      return new dictionary_coding_budget_optimizer_greedy_grow(*this);
    }

    void dictionary_coding_budget_optimizer_greedy_grow::optimize_in(std::vector<std::size_t>&  gamma_offset,
								std::vector<std::size_t>&  gamma_sz,
								std::vector<float>&        gamma_val,
								std::vector<std::size_t>&  gamma_idx,
								const std::vector<Eigen::VectorXf>& Y) {
      
      // Build multiple choice knapsack problem
      std::size_t Signal_Count = Y.size();
      std::size_t Signal_Dim  = Y[0].rows();
      std::size_t MCK_Slot_Count = Signal_Count / group_size_;
      if (MCK_Slot_Count * group_size_ < Signal_Count) ++MCK_Slot_Count; // Last slot partially empty!
      std::size_t MCK_Capacity = non_zeros_per_signal_ * MCK_Slot_Count;
      std::size_t MCK_Max_Non_Zero_Count = std::min(std::min(MCK_Capacity,max_non_zeros_per_signal_),
						    std::min(Signal_Dim,32*non_zeros_per_signal_)); 

      // Create incremental orthogonal matching pursuit optimizers
      std::vector<incremental_sparse_coder*> iomp(Signal_Count);
      std::vector<std::size_t> solution_nzc(MCK_Slot_Count);
      std::vector<double>      solution_err2(MCK_Slot_Count);
      std::vector<double>      solution_err2_up(MCK_Slot_Count);
#pragma omp parallel for
      for (std::size_t g=0; g<MCK_Slot_Count; ++g) {
	std::size_t i_bgn = g*group_size_;
	std::size_t i_end = std::min(Signal_Count, i_bgn+group_size_);
	solution_nzc[g] = 0;
	solution_err2[g] = 0.0;
	solution_err2_up[g] = 0.0;
	for (std::size_t i=i_bgn; i<i_end; ++i) {
	  iomp[i] = coder_->new_incremental_sparse_coder(Y[i]);
	  solution_err2[g] += iomp[i]->err2(solution_nzc[g]);
	  solution_err2_up[g] += iomp[i]->err2(solution_nzc[g]+1);
	}
      }

      std::size_t current_size=0;
      bool done=false;
      while (!done) {
	// Find best increase in error
	std::size_t best_g = std::size_t(-1);
	double best_delta = 0.0;
	for (std::size_t g=0; g<MCK_Slot_Count; ++g) {
	  if (solution_nzc[g]<MCK_Max_Non_Zero_Count) {
	    if (is_maxe2_reduction_enabled_) {
	      // Select largest error
	      double delta_g = solution_err2[g];
	      if (delta_g>best_delta) {
		// Select largest error
		best_g=g; best_delta=delta_g;
	      }
	      
	    } else {
	      double delta_g = solution_err2_up[g]-solution_err2[g];
	      if (delta_g<best_delta) {
		// Select largest decrease in error
		best_g=g; best_delta=delta_g;
	      }
	    }
	  }
	}
	if (best_g==std::size_t(-1)) {
	  // Cannot find improvement
	  done = true; 
	} else {
	  // Increase nonzero count of best group
	  ++current_size;
	  ++solution_nzc[best_g];
	  solution_err2[best_g] = solution_err2_up[best_g];
	  solution_err2_up[best_g] = 0.0;
	  std::size_t i_bgn = best_g*group_size_;
	  std::size_t i_end = std::min(Signal_Count, i_bgn+group_size_);
	  for (std::size_t i=i_bgn; i<i_end; ++i) {
	    solution_err2_up[best_g] += iomp[i]->err2(solution_nzc[best_g]+1);
	  }
	  done = (current_size==MCK_Capacity);
	}
      }
      
      // Allocate and store solution
      gamma_offset.clear(); gamma_offset.reserve(Signal_Count);
      gamma_sz.clear();     gamma_sz.reserve(Signal_Count);
      gamma_val.clear();    gamma_val.reserve(non_zeros_per_signal_*Signal_Count);
      gamma_idx.clear();    gamma_idx.reserve(non_zeros_per_signal_*Signal_Count);
      for (std::size_t g=0; g<MCK_Slot_Count; ++g) {
	std::size_t i_bgn = g*group_size_;
	std::size_t i_end = std::min(Signal_Count, i_bgn+group_size_);
	for (std::size_t i=i_bgn; i<i_end; ++i) {
	  const std::size_t k_i = solution_nzc[g]; 
	  const std::size_t offset_i = gamma_val.size();
	  gamma_sz.push_back(k_i);
	  gamma_offset.push_back(offset_i);
	  gamma_val.resize(offset_i+k_i);
	  gamma_idx.resize(offset_i+k_i);
	  if (k_i) {
	    iomp[i]->gamma_val_in(&(gamma_val[offset_i]), k_i);
	    iomp[i]->gamma_idx_in(&(gamma_idx[offset_i]), k_i);
	  }
	}
      }

      // Cleanup
      for (std::size_t i=0; i<Signal_Count; ++i) {
	delete iomp[i]; iomp[i] = 0;
      }
      iomp.clear();
    }

  } // namespace tdm_sparse_coding
} // namespace vic
