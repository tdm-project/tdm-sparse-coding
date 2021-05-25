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
#ifndef VIC_TDM_SPARSE_CODING_DICTIONARY_CODING_BUDGET_OPTIMIZER_GREEDY_GROW_HPP
#define VIC_TDM_SPARSE_CODING_DICTIONARY_CODING_BUDGET_OPTIMIZER_GREEDY_GROW_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sl/cstdint.hpp>
#include <sl/clock.hpp>
#include <vector>
#include <vic/tdm_sparse_coding/dictionary_coding_budget_optimizer.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {

    /**
     *  Budget optimizer using a greedy growing strategy exploting
     *  incremental orthogonal matching pursuit
     */
    class dictionary_coding_budget_optimizer_greedy_grow: public dictionary_coding_budget_optimizer {
    protected:
      bool is_maxe2_reduction_enabled_;
      
    public: // Creation & destruction
      
      inline dictionary_coding_budget_optimizer_greedy_grow(): is_maxe2_reduction_enabled_(false) {
      }

      virtual ~dictionary_coding_budget_optimizer_greedy_grow() {
      }

      virtual dictionary_coding_budget_optimizer_greedy_grow* clone() const;

      void set_is_maxe2_reduction_enabled(bool x) { is_maxe2_reduction_enabled_ = x; }

      bool is_maxe2_reduction_enabled() const { return is_maxe2_reduction_enabled_; }

    public: // Params

      virtual void optimize_in(std::vector<std::size_t>&  gamma_offset,
			       std::vector<std::size_t>&  gamma_sz,
			       std::vector<float>&        gamma_val,
			       std::vector<std::size_t>&  gamma_idx,
			       const std::vector<Eigen::VectorXf>& Y);

    }; // class dictionary_coding_budget_optimizer_greedy_grow
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
