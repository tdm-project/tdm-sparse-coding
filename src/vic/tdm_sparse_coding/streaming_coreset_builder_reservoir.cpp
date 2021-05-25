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
#include <vic/tdm_sparse_coding/streaming_coreset_builder_reservoir.hpp>
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

    streaming_coreset_builder_reservoir::streaming_coreset_builder_reservoir() {
      method_name_="reservoir";
      pass_count_ = 1;
      item_count_ = 0;
      w_sum_ = 0.0;
    }

    streaming_coreset_builder_reservoir::~streaming_coreset_builder_reservoir() {
    }
    

    void streaming_coreset_builder_reservoir::clear() {
      super_t::clear();
      pq_ = pq_t();
      item_count_ = 0;
      w_sum_ = 0.0;
    }
    
    void streaming_coreset_builder_reservoir::begin() {
      super_t::begin();
      //std::cerr << "BEGIN" << out_Y_.size() << std::endl;
      rng_.set_seed(desired_coreset_size_);
    }
      
    void streaming_coreset_builder_reservoir::end() {
      assert(pass_count_ == 1);
      assert(current_pass_ == 0);

      // Pr(y) = e2(y)/sum_Y e2(y)
      // Weight = 1/(C*Pr(y)) = sum_e2(y)/C/e2(y)

      const std::size_t C = pq_.size();
      if (C) {
	out_w_.clear(); out_w_.reserve(C);
	out_Y_.clear(); out_Y_.reserve(C);
	while(!pq_.empty()) {
	  out_Y_.push_back(pq_.top().second);
	  out_w_.push_back(w_sum_/e2(pq_.top().second)/C);
	  pq_.pop();
	}
      }
      
      // Clear
      w_sum_ = 0.0;
      item_count_ = 0;
      super_t::end();
    }
    
    void streaming_coreset_builder_reservoir::inner_put(const Eigen::VectorXf& y) {
      // Algorithm A-RES -- https://en.wikipedia.org/wiki/Reservoir_sampling
      double  e2_y = e2(y);
      if (e2_y<1e-7f) {
	++zero_y_count_;
      } else {
	++item_count_;
	double w_y = e2_y; w_sum_ += e2_y;
	float r_y = std::pow(rng_.value(), 1.0/w_y);
	if (pq_.size()<desired_coreset_size_) {
	  pq_.push(std::make_pair(r_y,y));
	} else if (pq_.top().first<r_y) {
	  pq_.pop();
	  pq_.push(std::make_pair(r_y,y));
	}
      }
    }
    
  } // namespace tdm_sparse_coding
} // namespace vic
