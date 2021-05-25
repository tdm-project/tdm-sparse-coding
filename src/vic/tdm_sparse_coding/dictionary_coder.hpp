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
#ifndef VIC_TDM_SPARSE_CODING_DICTIONARY_CODER_HPP
#define VIC_TDM_SPARSE_CODING_DICTIONARY_CODER_HPP

#include <vic/tdm_sparse_coding/signal_stream.hpp>
#include <vic/tdm_sparse_coding/incremental_sparse_coder.hpp>
#include <sl/random.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {

    class dictionary_coder {
    protected:
      // Common parameters for all implementations
      std::size_t   dim_;                        // Input data dimension (1d,2d,3d,..)
      std::size_t  Ndim_;                        // Signal size on each dimension
      std::size_t     N_;                        // Total signal size N = Ndim^dim
      std::size_t     K_;                        // Number of dictionary elements
      std::size_t     S_;                        // Max nonzero coefficients <= K (fixed rate + max for variable rate)
      float           T_;                        // Tolerance (variable rate encoding)

      // Hack -- refinement forced -- should be in a pursuit subclass...
      bool         is_omp_refining_enabled_;    // If true, perform refinement of omp solution
      
      // Dictionary - common to all implementations
      Eigen::MatrixXf D_;                           // N x K
      mutable Eigen::MatrixXf Dt_;                  // K x N
      mutable Eigen::MatrixXf DtD_;                 // K x K
      mutable bool            is_precomputed_coding_data_valid_; // true iff DtD is valid

      // Training
      std::size_t     training_epoch_count_;     // Training passes over coreset for iterative methods
      sl::uint64_t    max_coreset_scalar_count_; // Number of floats by default for methods using coreset

      // Random number generators
      sl::random::uniform_closed<float> randf_;
      sl::random::std_irng_t randi_;

    public:

      dictionary_coder();
      
      virtual ~dictionary_coder();

      virtual dictionary_coder* clone() const = 0;
      
    public: // Dictionary
      
      const Eigen::MatrixXf& dictionary() const { return D_; }

    public: // General parameters

      float tolerance() const { return T_; }

      virtual void set_tolerance(float x) { T_ = x; }

      std::size_t max_non_zeros() const { return S_; }
      //virtual void set_max_non_zeros(std::size_t x) { S_ = x; }

      std::size_t signal_dimension() const { return dim_; }

      std::size_t signal_size_per_dimension() const { return Ndim_; }
      
      std::size_t signal_size() const { return N_; }

      std::size_t dictionary_element_count() const { return K_; }

      bool is_omp_refining_enabled() const { return is_omp_refining_enabled_; }
      
    public: // Training interface

      sl::uint64_t max_coreset_scalar_count() const { return max_coreset_scalar_count_; }
      void set_max_coreset_scalar_count(sl::uint64_t x) { max_coreset_scalar_count_ = x; }
 
      std::size_t training_epoch_count() const { return training_epoch_count_; }
      void set_training_epoch_count(std::size_t n) { training_epoch_count_ = n; }
      
      virtual void reset() = 0;

      virtual void train(signal_stream& Y); 

      virtual void train_on_coreset(const std::vector<float>& W,
				    const std::vector<Eigen::VectorXf>& Y) = 0;

    public: // Reconstruction interface

      inline bool  is_precomputed_coding_data_valid() const { return is_precomputed_coding_data_valid_; }
      
      virtual void invalidate_precomputed_coding_data() const { is_precomputed_coding_data_valid_ = false; }

      // Call this *explicitly* before (parallel) decoding! 
      virtual void precompute_coding_data() {
	if (!is_precomputed_coding_data_valid()) {
	  Dt_ = D_.transpose();
	  DtD_ = Dt_*D_;
	  is_precomputed_coding_data_valid_ = true;
	}
      }
      
      // Encoding -- by default implemented through orthogonal matching pursuit
      // on dictionary -- subclasses may redefine method
      // NOTE: Requires is_precomputed_coding_data_valid() !!
      virtual void encode_in(std::size_t *gamma_idx,
			     float *gamma_val,
			     const Eigen::VectorXf& y,
			     std::size_t S = std::size_t(-1),	// max non zeros
			     std::size_t* out_S =0, // Achieved non zero coefficients
			     float* out_err2 = 0,  // Achieved error squared
			     float* out_err2_k = 0) const; // size K+1, on output is the sq error for 0..out_K
      
      // Decoding -- by default implemented through weighted sum of dictionary elements --
      // subclasses may redefine method
      virtual void decode_in(Eigen::VectorXf& y_tilde,
			     const std::size_t *gamma_idx,
			     const float *gamma_val,
			     std::size_t S=std::size_t(-1)) const;

      inline void approximate_in(Eigen::VectorXf& y_tilde,
				 const Eigen::VectorXf& y,
				 std::size_t S=std::size_t(-1)) const {
	if (S == std::size_t(-1)) S = S_;
	std::vector<std::size_t> gamma_idx(S);
	std::vector<float>       gamma_val(S);
	encode_in(&(gamma_idx[0]), &(gamma_val[0]), y, S);
	decode_in(y_tilde, &(gamma_idx[0]), &(gamma_val[0]), S);
      }
      
      void update_errors_in(float* out_err2,  
			    float* out_err2_k,
			    const std::size_t *gamma_idx,
			    const float* gamma_val,
			    const std::size_t S,
			    const Eigen::VectorXf& y) const;

    public: // sparse coder

      // Encoding -- by default implemented through orthogonal matching pursuit
      // on dictionary -- subclasses may redefine method
      // NOTE: Requires is_precomputed_coding_data_valid() !!
      virtual incremental_sparse_coder* new_incremental_sparse_coder(const Eigen::VectorXf& y) const;
      
    public: // Helpers

      static float coherence(const Eigen::MatrixXf& D);
      
      static float coherence_except(const Eigen::MatrixXf& D, std::size_t kk, const Eigen::VectorXf& y);
    
      static inline Eigen::VectorXf zero_mean(const Eigen::VectorXf& y) {
	// Remove mean from vector
	float y_avg = y.sum()/float(y.size());
	Eigen::VectorXf dy = y; dy.array() -= y_avg;
	return dy;
      }

    public: // Useful dictionary init stuff
      
      static void dictionary_init_random_dictionary_in(Eigen::MatrixXf& D0,
						       std::size_t N,
						       std::size_t K);
      
      static void dictionary_init_random_samples_in(Eigen::MatrixXf& D0,
						    const std::vector<Eigen::VectorXf>& Y,
						    std::size_t K);
      static void dictionary_init_coarse_fine_separated_samples_in(Eigen::MatrixXf& D0,
								   const std::vector<Eigen::VectorXf>& Y,
								   std::size_t dim,
								   std::size_t Ndim,
								   std::size_t K);
      static void dictionary_init_separated_samples_in(Eigen::MatrixXf& D0,
						       const std::vector<Eigen::VectorXf>& Y,
						       std::size_t K);

      static void dictionary_init_in(Eigen::MatrixXf& D0,
				     const std::vector<Eigen::VectorXf>& Y,
				     std::size_t dim,
				     std::size_t Ndim,
				     std::size_t K,
				     float max_coherence=1.0f);

      static void dictionary_decorrelate_in(Eigen::MatrixXf& D,
					    float max_coherence);

    };
    
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
