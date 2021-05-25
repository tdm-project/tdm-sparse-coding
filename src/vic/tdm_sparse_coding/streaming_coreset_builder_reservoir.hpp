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
#ifndef VIC_TDM_SPARSE_CODING_STREAMING_CORESET_BUILDER_RESERVOIR_HPP
#define VIC_TDM_SPARSE_CODING_STREAMING_CORESET_BUILDER_RESERVOIR_HPP

#include <vic/tdm_sparse_coding/streaming_coreset_builder.hpp>
#include <sl/random.hpp>
#include <queue>

/*

  From
    Feldman, Dan, Micha Feigin, and Nir Sochen.
    "Learning big (image) data via coresets for dictionaries."
    Journal of mathematical imaging and vision 46.3 (2013): 276-291.

  Error estimate: e2(y) = e2(y_avg + dy) ~ e2(dy) ~ ||dy||^2

  Weight ~ Error estimate
  
  Pr(y) = e2(y)/sum_Y e2(y)
  Weight = 1/(C*Pr(y)) = sum_e2(y)/C * 1/e2(y)
  
*/
  
namespace vic {
  
  namespace tdm_sparse_coding {
    
    /*
    ** Streaming coreset builder based on a simple reservoir
    ** sampling strategy. 
    **
    ** Reservoir sampling strategy approximately from 
    **   Efraimidis, Pavlos S.; Spirakis, Paul G. (2006-03-16).
    **   "Weighted random sampling with a reservoir".
    **   Information Processing Letters. 97 (5): 181–185.
    **
    ** See also
    **   Chao, M.T. (1982) A general purpose unequal probability
    **   sampling plan. Biometrika, 69 (3): 653-656.
    **
    ** Picking probabilities and coreset weighting:
    **   Feldman, Dan, Micha Feigin, and Nir Sochen.
    **   Learning big (image) data via coresets for dictionaries.
    **   Journal of mathematical imaging and vision 46.3 (2013): 276-291.
    */
    class streaming_coreset_builder_reservoir: public streaming_coreset_builder {
    protected:
      typedef streaming_coreset_builder super_t;

      typedef std::pair<float, Eigen::VectorXf> pv_pair_t;

      struct pv_pair_gt {
	bool operator()(const pv_pair_t& lhs, const pv_pair_t& rhs) const {
	  if (lhs.first>rhs.first) {
	    return true;
	  } else if (lhs.first<rhs.first) {
	    return false;
	  } else {
	    // first equal
	    for (int i= 0; i< rhs.second.size(); i++) {
	      if (lhs.second[i] > rhs.second[i]) {
		return true;
	      } else if (lhs.second[i] < rhs.second[i]) {
		return false;
	      }
	    }
	    // first and second equal
	    return false;
	  }
	}
      };

      typedef std::priority_queue< pv_pair_t, std::vector<pv_pair_t>, pv_pair_gt> pq_t;
      
    protected:
      sl::random::uniform<float> rng_;
      pq_t pq_;

      sl::uint64_t    item_count_;
      double          w_sum_;
      
    public: // Creation & destruction
      
      streaming_coreset_builder_reservoir();

      virtual ~streaming_coreset_builder_reservoir();

      virtual void clear();

    public: // Params
     
    public: // Output

    public: // Input: Streaming interface
      
      virtual void begin();
      
      virtual void end();

    protected:

      virtual void inner_put(const Eigen::VectorXf& y);
      
    }; // class streaming_coreset_builder_reservoir
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
