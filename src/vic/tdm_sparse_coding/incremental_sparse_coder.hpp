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
#ifndef VIC_TDM_SPARSE_CODING_INCREMENTAL_SPARSE_CODER_HPP
#define VIC_TDM_SPARSE_CODING_INCREMENTAL_SPARSE_CODER_HPP

#include <vic/tdm_sparse_coding/matching_pursuit.hpp>
#include <cassert>
#include <sl/math.hpp>

namespace vic {

  namespace tdm_sparse_coding {

    /**
     * Base class for incremental sparse coders
     */
    class incremental_sparse_coder {
    protected:
      // Incremental output
      std::vector<std::size_t> gamma_offset_;
      std::vector<float> gamma_val_;
      std::vector<std::size_t> gamma_idx_;
      std::vector<float> err2_;
    public:

      incremental_sparse_coder() {
      }

      virtual ~incremental_sparse_coder() {
      }

      virtual void clear();
      
    public:

      virtual bool is_valid() const = 0;
      
      inline std::size_t max_computed_nonzero_count() const {
	return err2_.size()-1; // Will be std::size_t(-1) if not initialized
      }
      
      inline float err2(std::size_t S) {
	assert(is_valid());
	update_to(S); return err2_[S];
      }
      
      inline void gamma_val_in(float* gamma_val, std::size_t S) {
	assert(S>0);
	assert(is_valid());
	assert(gamma_val);
	update_to(S);
	const std::size_t offset_s = gamma_offset_[S-1];
	for (std::size_t s=0; s<S; ++s) {
	  gamma_val[s] = gamma_val_[offset_s+s];
	}
      }
      
      inline void gamma_idx_in(std::size_t* gamma_idx, std::size_t S) {
	assert(S>0);
	assert(is_valid());
	assert(gamma_idx);
	update_to(S);
	const std::size_t offset_s = gamma_offset_[S-1];
	for (std::size_t s=0; s<S; ++s) {
	  gamma_idx[s] = gamma_idx_[offset_s+s];
	}
      }

    protected:

      virtual void update_to(std::size_t S) = 0;
    };
      
  }
  
}

#endif
