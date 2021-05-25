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
#ifndef VIC_TDM_SPARSE_CODING_DICTIONARY_CODER_KSVD_HPP
#define VIC_TDM_SPARSE_CODING_DICTIONARY_CODER_KSVD_HPP

#include <vic/tdm_sparse_coding/dictionary_coder.hpp>
#include <vic/tdm_sparse_coding/error_evaluator.hpp>

namespace vic {

  namespace tdm_sparse_coding {
        

    /*
      ----------------------------------------------------------------------
      DICTIONARY_CODER_KSVD
      
      References:
      
      [1] M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm
      for Designing of Overcomplete Dictionaries for Sparse
      Representation", the IEEE Trans. On Signal Processing, Vol. 54, no.
      11, pp. 4311-4322, November 2006.
      
      [2] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation
      of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit",
      Technical Report - CS, Technion, April 2008.
      ----------------------------------------------------------------------
    */
    class dictionary_coder_ksvd: public dictionary_coder {
    protected:
      error_evaluator last_stats_;
      float max_coherence_;
      std::size_t dictionary_update_cycle_count_;
      bool is_l1_pca_enabled_;

      
      Eigen::MatrixXf D_init_;
      
    public:

      dictionary_coder_ksvd();
      
      virtual ~dictionary_coder_ksvd();

      virtual dictionary_coder_ksvd* clone() const;

    public: //

      inline void set_init_dictionary(const Eigen::MatrixXf& D) {
	D_init_ = D;
      }

      inline void reset_init_dictionary() {
	D_init_ = Eigen::MatrixXf();
      }

      inline bool is_using_precomputed_init_dictionary() const {
	return D_init_.size() != 0;
      }
      
    public: // coder interface

      virtual void reset();
      
      virtual void train_on_coreset(const std::vector<float>&  W_subset,
				    const std::vector<Eigen::VectorXf>& Y_subset);

    public: // Specific interface
      
      void init(std::size_t dim, std::size_t Ndim, std::size_t K, std::size_t S, float T,
		float max_coh, std::size_t dictionary_update_cycle_count= 1, bool is_l1_pca_enabled_ = false);

    protected: // Implementation
      
      static void compressed_rows_in(std::vector<std::vector<std::size_t> >& I_x,
				     std::vector<std::vector<float> >& gamma_j,
				     std::vector<std::vector<std::size_t> >& gamma_j_offset,
				     const std::size_t N,
				     const std::size_t M,
				     const std::size_t column_to_skip,
				     const std::vector<std::size_t>& gamma_offset,
				     const std::vector<std::size_t>& gamma_sz,
				     const std::vector<float>&       gamma_val,
				     const std::vector<std::size_t>& gamma_idx);
      static void unused_atom_in(Eigen::VectorXf& atom,
				 std::vector<bool>& is_atom_replaced,
				 std::vector<bool>& is_signal_used,
				 std::size_t jj,
				 const Eigen::MatrixXf& D,
				 const std::vector<float>& W,
				 const std::vector<Eigen::VectorXf>& Y,
				 const std::vector<std::size_t>& gamma_offset,
				 const std::vector<std::size_t>& gamma_sz,
				 const std::vector<float>&       gamma_val,
				 const std::vector<std::size_t>& gamma_idx);

      static void improve_atom_l1_in(Eigen::VectorXf& atom,
				     const std::vector<float>& gamma_j,
				     const std::vector<std::size_t>& I_x,
				     std::size_t jj,
				     const Eigen::MatrixXf& D,
				     const std::vector<float>& W,
				     const  std::vector<Eigen::VectorXf>& Y,
				     const std::vector<std::size_t>& gamma_offset,
				     const std::vector<std::size_t>& gamma_sz,
				     const std::vector<float>&       gamma_val,
				     const std::vector<std::size_t>& gamma_idx);

      static void improve_atom_l2_in(Eigen::VectorXf& atom,
				     const std::vector<float>& gamma_j,
				     const std::vector<std::size_t>& I_x,
				     std::size_t jj,
				     const Eigen::MatrixXf& D,
				     const std::vector<float>& W,
				     const  std::vector<Eigen::VectorXf>& Y,
				     const std::vector<std::size_t>& gamma_offset,
				     const std::vector<std::size_t>& gamma_sz,
				     const std::vector<float>&       gamma_val,
				     const std::vector<std::size_t>& gamma_idx);
      
      static void improve_gamma_in(std::vector<float>& gamma_j,
				   const Eigen::VectorXf& atom,
				   const std::vector<std::size_t>& I_x,
				   std::size_t jj,
				   const Eigen::MatrixXf& D,
				   const std::vector<float>& W,
				   const  std::vector<Eigen::VectorXf>& Y,
				   const std::vector<std::size_t>& gamma_offset,
				   const std::vector<std::size_t>& gamma_sz,
				   const std::vector<float>&       gamma_val,
				   const std::vector<std::size_t>& gamma_idx);
      
      static void dictionary_cleanup_in(Eigen::MatrixXf& D,
					std::vector<bool>& is_signal_used,
					const std::vector<bool>& is_atom_replaced,
					const std::vector<float>& W,
					const std::vector<Eigen::VectorXf>& Y,
					const std::vector<std::size_t>& gamma_offset,
					const std::vector<std::size_t>& gamma_sz,
					const std::vector<float>&       gamma_val,
					const std::vector<std::size_t>& gamma_idx,
					float mutual_incoherence_threshold = 0.99f,
					std::size_t use_threshold = 4);
      
     static void refine_in(Eigen::MatrixXf& D,
			   error_evaluator& refine_stats,
			   const  std::vector<float>& W,
			   const  std::vector<Eigen::VectorXf>& Y,
			   std::size_t K,
			   float tol,
			   float max_coh,
			   std::size_t dictionary_update_cycle_count,
			   bool is_l1_pca_enabled,
			   std::size_t epoch_count);
    };
  } // namespace sparse_coding

} // namespace vic

#endif
  
