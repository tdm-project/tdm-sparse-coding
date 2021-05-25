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
#ifndef VIC_TDM_SPARSE_CODING_INCREMENTAL_SPARSE_CODER_OMP_HPP
#define VIC_TDM_SPARSE_CODING_INCREMENTAL_SPARSE_CODER_OMP_HPP

#include <vic/tdm_sparse_coding/incremental_sparse_coder.hpp>
#include <cassert>
#include <sl/math.hpp>

namespace vic {

  namespace tdm_sparse_coding {

    class incremental_sparse_coder_omp: public incremental_sparse_coder {
    public:
      typedef incremental_sparse_coder_omp this_t;
      typedef incremental_sparse_coder super_t;
    protected:
      // Input
      Eigen::VectorXf  Dty_;
      const Eigen::MatrixXf* DtD_ptr_;

      // Internal state
      double          state_deltaprev_;
      Eigen::VectorXf state_alpha_;
      Eigen::MatrixXf state_Lchol_;

    public:

      incremental_sparse_coder_omp(): DtD_ptr_(0), state_deltaprev_(0.0) {
      }

      virtual ~incremental_sparse_coder_omp() {
      }

      virtual void clear();
      
      void init(const Eigen::MatrixXf* DtD_ptr,
		const Eigen::VectorXf& Dty,
		float yty);

    public:
      
      inline virtual bool is_valid() const {
	return DtD_ptr_ != 0;
      }

    protected:
      
      inline virtual void update_to(std::size_t S) {
	assert(is_valid());

	while (max_computed_nonzero_count()<S) step();

	assert(err2_.size()>=S+1);
	assert(gamma_val_.size()>=S);
	assert(gamma_idx_.size()>=S);
      }
      
      void step();
      void null_step();
      void clear_state();
    };
      
  }
  
}

#endif
