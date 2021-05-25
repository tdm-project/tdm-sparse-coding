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
#include <vic/tdm_sparse_coding/streaming_coreset_builder_uniform.hpp>
#include <sl/utility.hpp>

namespace vic {
  namespace tdm_sparse_coding {

    // -- Supporting routines
    
    static inline Eigen::VectorXf zero_mean(const Eigen::VectorXf& y) {
      // Remove mean from vector
      float y_avg = y.sum()/float(y.size());
      Eigen::VectorXf dy = y; dy.array() -= y_avg;
      return dy;
    }
    
    static inline float e2(const Eigen::VectorXf& y) {
      return zero_mean(y).squaredNorm();
    }

    // Implementation

    streaming_coreset_builder_uniform::streaming_coreset_builder_uniform() {
      method_name_="uniform";
      pass_count_ = 1;
      item_count_ = 0;
    }

    streaming_coreset_builder_uniform::~streaming_coreset_builder_uniform() {
    }
    

    void streaming_coreset_builder_uniform::clear() {
      super_t::clear();
      item_count_ = 0;
    }
    
    void streaming_coreset_builder_uniform::begin() {
      super_t::begin();
      //std::cerr << "BEGIN" << out_Y_.size() << std::endl;
      irng_.set_seed(desired_coreset_size_);
    }
      
    void streaming_coreset_builder_uniform::end() {
      assert(pass_count_ == 1);
      assert(current_pass_ == 0);

      // Adjust weights
      double C = double(out_w_.size());
      for (std::size_t i=0; i<out_w_.size(); ++i) {
	out_w_[i] /= C;
      }
      
      // Clear
      item_count_ = 0;
      super_t::end();
    }
    
    void streaming_coreset_builder_uniform::inner_put(const Eigen::VectorXf& y) {
      // https://en.wikipedia.org/wiki/Reservoir_sampling
      double e2_y = e2(y);
      if (e2_y>1e-7f) {
	++item_count_;
	const std::size_t R=out_Y_.size();
	if (R < desired_coreset_size_) {
	  const double picking_probability = 1.0;
	  out_w_.push_back(1.0);
	  out_Y_.push_back(y);
	} else {
	  const std::size_t r = irng_.value() % item_count_;
	  if (r<desired_coreset_size_) {
	    out_w_[r] = 1.0;
	    out_Y_[r] = y;
	  }
	}
      } else {
        ++zero_y_count_;
      }
    }
    
  } // namespace tdm_sparse_coding
} // namespace vic
