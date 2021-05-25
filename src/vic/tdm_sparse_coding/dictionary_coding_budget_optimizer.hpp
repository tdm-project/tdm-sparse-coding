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
#ifndef VIC_TDM_SPARSE_CODING_DICTIONARY_CODING_BUDGET_OPTIMIZER_HPP
#define VIC_TDM_SPARSE_CODING_DICTIONARY_CODING_BUDGET_OPTIMIZER_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sl/cstdint.hpp>
#include <sl/clock.hpp>
#include <vector>
#include <vic/tdm_sparse_coding/dictionary_coder.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {

    /**
     *  Single pass streaming coreset builder
     */
    class dictionary_coding_budget_optimizer {
    protected:
      std::string     method_name_;
      bool            is_verbose_;
      std::size_t     non_zeros_per_signal_;
      std::size_t     max_non_zeros_per_signal_;
      std::size_t     group_size_;

      dictionary_coder* coder_;
      
    public: // Creation & destruction
      
      inline dictionary_coding_budget_optimizer():
	method_name_("none"),
	is_verbose_(true),
	non_zeros_per_signal_(0),
	max_non_zeros_per_signal_(std::size_t(-1)),
	group_size_(1),
	coder_(0) {
      }

      virtual ~dictionary_coding_budget_optimizer() {}

      virtual dictionary_coding_budget_optimizer* clone() const;

    public: // Params

      inline bool is_verbose() const { return is_verbose_; }
      inline void set_is_verbose(bool x) { is_verbose_ = x; }

      // Desidered average for variable rate encoders
      inline std::size_t non_zeros_per_signal() const { return non_zeros_per_signal_; }
      inline void set_non_zeros_per_signal(std::size_t S) { non_zeros_per_signal_ = S; }

      // Upper bound for variable rate encoders - (-1) if no limit
      inline std::size_t max_non_zeros_per_signal() const { return max_non_zeros_per_signal_; }
      inline void set_max_non_zeros_per_signal(std::size_t S) { max_non_zeros_per_signal_ = S; }

      // Number of consecutive signal that should share the same non zero count
      inline std::size_t group_size() const { return group_size_; }
      inline void set_group_size(std::size_t G) { group_size_ = G; }
     
      inline dictionary_coder* coder() { return coder_; }
      inline void set_coder(dictionary_coder* c) { coder_ = c; }

    public:

      virtual void optimize_in(std::vector<std::size_t>&  gamma_offset,
			       std::vector<std::size_t>&  gamma_sz,
			       std::vector<float>&        gamma_val,
			       std::vector<std::size_t>&  gamma_idx,
			       const std::vector<Eigen::VectorXf>& Y);

    public: // Helpers
      
      void decode_in(Eigen::VectorXf& y_tilde,
		     const std::size_t *gamma_idx,
		     const float *gamma_val,
		     std::size_t S) const;

      // Encode and decode - useful for error evaluation
      virtual void approximate_in(std::vector<Eigen::VectorXf>& Y_tilde,
				  const std::vector<Eigen::VectorXf>& Y,
				  std::size_t* gamma_count);

    }; // class dictionary_coding_budget_optimizer
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
