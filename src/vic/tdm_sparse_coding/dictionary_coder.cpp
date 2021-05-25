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
#include <vic/tdm_sparse_coding/dictionary_coder.hpp>
#include <vic/tdm_sparse_coding/matching_pursuit.hpp>
#include <vic/tdm_sparse_coding/incremental_sparse_coder_omp.hpp>
#include <vic/tdm_sparse_coding/streaming_coreset_builder_reservoir.hpp>
#include <sl/random.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {
    
    dictionary_coder::dictionary_coder() :
      dim_(1),
      Ndim_(0),
      N_(0),
      K_(0),
      S_(0),
      T_(0.0f),
      is_precomputed_coding_data_valid_(false),
      training_epoch_count_(100),
      max_coreset_scalar_count_(32*1024*1024) {
    }

    dictionary_coder::~dictionary_coder() {
    }

    void dictionary_coder::train(signal_stream& Y) {
      // Default implementation
      std::cerr << "----------------------------------------------------------" << std::endl;
      std::cerr << "CORESET EXTRACTION" << std::endl;
      std::cerr << "----------------------------------------------------------" << std::endl;
      std::vector<float>           W_sub;
      std::vector<Eigen::VectorXf> Y_sub;
      streaming_coreset_builder_reservoir scb;
      const std::size_t coreset_blocks = max_coreset_scalar_count_ / sl::ipow(Y.dimension(),3);
      scb.set_desired_coreset_size(coreset_blocks);
      scb.build(Y);
      scb.extract_weights_and_signals_in(W_sub,Y_sub);

      std::cerr << "----------------------------------------------------------" << std::endl;
      std::cerr << "TRAINING ON CORESET" << std::endl;
      std::cerr << "----------------------------------------------------------" << std::endl;
      train_on_coreset(W_sub,Y_sub);
    }

    void dictionary_coder::update_errors_in(float* out_err2,  
					    float* out_err2_k,
					    const std::size_t *gamma_idx,
					    const float* gamma_val,
					    const std::size_t S,
					    const Eigen::VectorXf& y) const {
      if (out_err2 || out_err2_k) {
	Eigen::VectorXf r = y;
	out_err2_k[0] = r.squaredNorm();
	for (std::size_t s=0; s<S; ++s) {
	  r -= gamma_val[s] * D_.col(gamma_idx[s]);
	  if (out_err2_k) out_err2_k[s+1] = r.squaredNorm();
	}	
	if (out_err2) *out_err2 = r.squaredNorm();
      }
    }
    
    void dictionary_coder::encode_in(std::size_t *gamma_idx,
				     float* gamma_val,
				     const Eigen::VectorXf& y,
				     std::size_t S,
				     std::size_t* out_S, 
				     float* out_err2,  
				     float* out_err2_k) const {
      assert(is_precomputed_coding_data_valid());

      omp_batch_in(gamma_val, gamma_idx,
		   Dt_*y,
		   DtD_,
		   (S == std::size_t(-1) ? S_ : S),
		   y.dot(y),
		   T_,
		   out_S, out_err2, out_err2_k);
      
      if (is_omp_refining_enabled_) {
	if ((*out_S) > 2) {
	  imp_refine_in(gamma_val, gamma_idx, *out_S,
			D_, y, Dt_ * y, DtD_);
	  if (out_err2 || out_err2_k) {
	    update_errors_in(out_err2, out_err2_k,
			     gamma_idx, gamma_val,
			     *out_S, y);
	  }
	}
      }
    }

    incremental_sparse_coder* dictionary_coder::new_incremental_sparse_coder(const Eigen::VectorXf& y) const {
      incremental_sparse_coder_omp* result = new incremental_sparse_coder_omp;
      result->init(&DtD_, Dt_*y, y.dot(y));
      return result;
    }

    void dictionary_coder::decode_in(Eigen::VectorXf& y_tilde,
				     const std::size_t *gamma_idx,
				     const float *gamma_val,
				     std::size_t S) const {
      if (S == std::size_t(-1)) S = S_;

      y_tilde.resize(N_);
      y_tilde.setConstant(0.0f);
      for (std::size_t k=0; k<S; ++k) {
	y_tilde += gamma_val[k] * D_.col(gamma_idx[k]);
      }
    }

    float dictionary_coder::coherence(const Eigen::MatrixXf& D) {
      float result = 0.0f;

      for (std::size_t k1=0;  k1<std::size_t(D.cols()); ++k1) {
	for (std::size_t k2=k1+1; k2<std::size_t(D.cols()); ++k2) {
	  if (k1!=k2) {
	    result = std::max(result, sl::abs(D.col(k1).dot(D.col(k2))));
	  }
	}
      }
      return result;
    }
      
    float dictionary_coder::coherence_except(const Eigen::MatrixXf& D, std::size_t kk, const Eigen::VectorXf& y) {
      float result = 0.0f;

      Eigen::VectorXf y1 = y; y1.normalize();
      for (std::size_t k=0; k<std::size_t(D.cols()); ++k) {
	if (k!=kk) {
	  result = std::max(result, sl::abs(y1.dot(D.col(k))));
	}
      }
      return result;
    }
      
    // =====================================================================================
    
    void dictionary_coder::dictionary_init_random_dictionary_in(Eigen::MatrixXf& D0,
								std::size_t N,
								std::size_t K) {

      std::cerr << "DICTIONARY INIT: PURE RANDOM" << std::endl;
      D0.resize(N,K);
      
      const float      eps = 1e-7f;
      Eigen::VectorXf d_k(N);
      for (std::size_t k=0; k<K; ++k) {
    	float d_k_norm = 0.0f;
    	if (k==0) {
    	  d_k.fill(1.0f);
    	  d_k_norm = d_k.norm();
    	} else {
    	  while (d_k_norm<eps) {
    	    d_k.setRandom();
	    d_k = zero_mean(d_k);
    	    d_k_norm = d_k.norm();
    	  }
    	}
    	D0.col(k) = (d_k/d_k_norm);
      }
      std::cerr << "COH = " << coherence(D0) << std::endl;
    }
    
    void dictionary_coder::dictionary_init_random_samples_in(Eigen::MatrixXf& D0,
							     const std::vector<Eigen::VectorXf>& Y,
							     std::size_t K) {
      std::cerr << "DICTIONARY INIT: RANDOMLY SELECTED SAMPLES" << std::endl;

      const std::size_t N=Y[0].rows();
      const std::size_t M=Y.size();
      D0.resize(N,K);
      
      std::vector<std::size_t> permutation(M);
      for (std::size_t k=0; k<M; ++k) {
	permutation[k] = k;
      }
      std::random_shuffle(permutation.begin(), permutation.end());

      std::size_t i = 0;
      for (std::size_t k=0; k<std::size_t(D0.cols()); ++k) {
	std::cerr << "INIT: " << k+1 << "/" << D0.cols() << "\r";
	
	const float      eps = N*(1e-6f);
	Eigen::VectorXf  y_k;
	float            y_k_norm = 0.0f;
	if (k==0) {
	  // First column is constant = 1
	  y_k.resize(N);
	  y_k.fill(1.0f);
	  y_k_norm = y_k.norm();
	  //std::cerr << "D[" << k << "] <- [1]" << std::endl; 
	} else {
	  // Try picking one vector from examples
	  while (i<M && y_k_norm<eps) {
	    std::size_t i_k = permutation[i];
	    y_k = zero_mean(Y[i_k]);
	    y_k_norm = y_k.norm();
	    if (y_k_norm>=eps) {
	      // Check that it is incoherent enough
	      bool is_linearly_dependent = false;
	      for (std::size_t k1=0; k1<k && (!is_linearly_dependent); ++k1) {
		is_linearly_dependent = (D0.col(k1).dot(y_k) > 0.99f*0.99f * y_k_norm);
	      }
	      if (is_linearly_dependent) y_k_norm=0.0f;
	    }
	    ++i;
	  }
	  // FIXME if (y_k_norm>=eps) std::cerr << "D[" << k << "] <- Y[" << permutation[i-1] << "]" << std::endl; 
	  // If not, pick random non-null vector
	  while (y_k_norm<eps) {
	    // FIXME std::cerr << "D[" << k << "] <- RANDOM!" << std::endl;
	    y_k.resize(N); y_k.setRandom();
	    y_k = zero_mean(y_k);
	    y_k_norm = y_k.norm();
	  }
	}
	D0.col(k) = (y_k/y_k_norm);
      }
      std::cerr << "COH = " << coherence(D0) << std::endl;
    } 

    void dictionary_coder::dictionary_init_separated_samples_in(Eigen::MatrixXf& D0,
								const std::vector<Eigen::VectorXf>& Y,
								std::size_t K) {
      std::cerr << "DICTIONARY INIT: SEPARATED SAMPLES" << std::endl;

      const std::size_t N=Y[0].rows();
      D0.resize(N,K);
      
      const std::size_t R=std::min(50*K,Y.size());
      std::vector<std::size_t> permutation(R);

      permutation.resize(Y.size());
      for (std::size_t k=0; k<Y.size(); ++k) {
	permutation[k] = k;
      }
      std::random_shuffle(permutation.begin(), permutation.end());
      permutation.resize(R);

      for (std::size_t k=0; k<K; ++k) {
	std::cerr << "INIT: " << k+1 << "/" << K << "\r";

	std::random_shuffle(permutation.begin(), permutation.end());
	
	const float      eps = 1e-7f;
	Eigen::VectorXf  d_k;
	float            d_k_norm = 0.0f;
	if (k==0) {
	  // Init first column of dictionary
	  d_k.resize(N);
	  d_k.fill(1.0f);
	  d_k_norm = d_k.norm(); 
	} else {
	  // std::cerr << std::endl;

	  // Add one column maximizing incoherence with current dictionary
	  const std::size_t trials=256;
	  std::size_t       m_k_best;
	  // -- FIXME std::size_t       m_k_near_best=0;
	  float             coh_best= 1.0f;
	  for (std::size_t i=0; i<std::min(permutation.size(),trials); ) {
	    std::size_t m_k = permutation[i];
	    Eigen::VectorXf y_k = zero_mean(Y[m_k]);	    
 	    float y_k_norm = y_k.norm();
	    if (y_k_norm<eps) {
	      // Not valid, remove
	      std::swap(permutation[i], permutation[permutation.size()-1]);
	      permutation.pop_back();
	    } else {
	      // Valid, try it
	      y_k /= y_k_norm;
	      float coh = 0.0;
	      //std::size_t nearest=0;
	      for (std::size_t k1=0; k1<k && coh<coh_best; ++k1) {
		float dot = y_k.dot(D0.col(k1));
		//std::cerr << k1 << ": " << dot << std::endl;
		float this_coh = sl::abs(dot);
		if (this_coh>coh) {
		  coh=this_coh;
		  //nearest=k1;
		}
	      }
	      //std::cerr << "coh[" << k << " = " << coh << "]" << std::endl;
	      if (coh<coh_best) {
		m_k_best = m_k;
		coh_best = coh;
		// FIXME -- m_k_near_best = nearest;
	      }
	      // If high coherence, remove from possible future candidates
	      if (coh > 0.9) {
		std::swap(permutation[i], permutation[permutation.size()-1]);
		permutation.pop_back();
	      } else {
		// Next
		++i;
	      }
	    }
	  }

	  // Assign to dictionary
	  
	  // First, try one with lowest coherence...	  
	  if (coh_best<1.0) {
	    d_k = zero_mean(Y[m_k_best]);
	    //-- FIXME d_k -= d_k.dot(D0.col(m_k_near_best))*D0.col(m_k_near_best);
	    d_k_norm = d_k.norm();
	    if (d_k_norm>=eps) {
	      //std::cerr << "Setting max separated: " << m_k_best << " " << coh_best << std::endl;
	    }
	  }
	  
	  // ... if not, try picking one vector from examples
	  while (!permutation.empty() && d_k_norm<eps) {
	    std::size_t m_k = permutation.back();
	    d_k = zero_mean(Y[m_k]);
	    d_k_norm = d_k.norm();
	    permutation.pop_back();
	    if (d_k_norm>=eps) {
	      std::cerr << "Setting random example: " << m_k << std::endl;
	    }
	  }
	  
	  // .. if not, pick random non-null vector
	  while (d_k_norm<eps) {
	    d_k.resize(N); d_k.setRandom();
	    d_k = zero_mean(d_k);
	    d_k_norm = d_k.norm();
	    if (d_k_norm>=eps) {
	      std::cerr << "Setting pure random: " << std::endl;
	    }
	  }
	} // if k

	// Append d_k to dictionary
	D0.col(k) = (d_k/d_k_norm);
      } // for k
      std::cerr << std::endl;

      // Generate output
      std::cerr << "COH = " << coherence(D0) << std::endl;
    }

    static void coarse_fine_in(Eigen::VectorXf& y_coarse,
			       Eigen::VectorXf& y_fine,
			       std::size_t dim,
			       std::size_t Ndim,
			       const Eigen::VectorXf& y_orig) {
      if ((dim!=3) || (Ndim%2)!=0) { //  FIXME temporarily implemented only for dim = 3 and even size
	y_coarse = y_orig;
      } else {
	std::size_t N = sl::ipow(Ndim,dim);
	y_coarse.resize(N);
	for (std::size_t z=0; z<Ndim/2; ++z) {
	  for (std::size_t y=0; y<Ndim/2; ++y) {
	    for (std::size_t x=0; x<Ndim/2; ++x) {
	      float avg=0.0f;
	      for (std::size_t dz=0; dz<2; ++dz) {
		for (std::size_t dy=0; dy<2; ++dy) {
		  for (std::size_t dx=0; dx<2; ++dx) {
		    const std::size_t idx = (x+dx)+Ndim*((y+dy)+Ndim*(z+dz));
		    avg += y_orig[idx];
		  }
		}
	      }
	      avg/=8.0f;
	      
	      for (std::size_t dz=0; dz<2; ++dz) {
		for (std::size_t dy=0; dy<2; ++dy) {
		  for (std::size_t dx=0; dx<2; ++dx) {
		    std::size_t idx = (x+dx)+Ndim*((y+dy)+Ndim*(z+dz));
		    y_coarse[idx] = avg;
		  }
		}
	      }
	    } // x
	  } // y
	} // z
      } // if 3d
      y_fine = y_orig - y_coarse; 
    }
    
    void dictionary_coder::dictionary_init_coarse_fine_separated_samples_in(Eigen::MatrixXf& D0,
									    const std::vector<Eigen::VectorXf>& Y,
									    std::size_t dim,
									    std::size_t Ndim,
									    std::size_t K) {
      if (dim!= 3 || (Ndim%2)!=0) { 
	dictionary_init_separated_samples_in(D0,Y,K);
      } else {
	std::cerr << "DICTIONARY INIT: COARSE-FINE SEPARATED SAMPLES" << std::endl;

	const std::size_t N=Y[0].rows();
	const std::size_t R=std::min(50*K,Y.size());
	sl::random::std_irng_t irng;
	
	D0.resize(N,K);
	
	std::vector<std::size_t> permutation(R);
	
	for (std::size_t k=0; k<K; ++k) {
	  std::cerr << "INIT COARSE: " << k+1 << "/" << K << "\r";

	  if (k==0 || k==K/8) { // Fist coarse or first fine
	    //Reservoir sampling --  Algorithm R by Jeffrey Vitter
	    for (std::size_t k=0; k<Y.size(); ++k) {
	      if (k<R) {
		permutation[k] = k;
	      } else {
		std::size_t M = irng.value() % (k+1);
		if (M<R) {
		  permutation[M] = k;
		}
	      }
	    }
      	  }
	  std::random_shuffle(permutation.begin(), permutation.end());
	  
	  const float      eps = 1e-7f;
	  Eigen::VectorXf  d_k;
	  float            d_k_norm = 0.0f;
	  if (k==0) {
	    // Init first column of dictionary
	    d_k.resize(N);
	    d_k.fill(1.0f);
	    d_k_norm = d_k.norm();
	  } else {
	    // std::cerr << std::endl;

	    // Add one column maximizing incoherence with current dictionary
	    const std::size_t trials=256;
	    std::size_t       m_k_best;
	    float             coh_best= 1.0f;
	    for (std::size_t i=0; i<std::min(permutation.size(),trials); ) {
	      std::size_t m_k = permutation[i];
	      Eigen::VectorXf y_k = zero_mean(Y[m_k]);
	      Eigen::VectorXf y_k_coarse, y_k_fine;
	      coarse_fine_in(y_k_coarse, y_k_fine, dim, Ndim, y_k);
	      if (k<K/8) { y_k=y_k_coarse; } else { y_k=y_k_fine; }
	      float y_k_norm = y_k.norm();
	      if (y_k_norm<eps) {
		// Not valid, remove
		std::swap(permutation[i], permutation[permutation.size()-1]);
		permutation.pop_back();
	      } else {
		// Valid, try it
		y_k /= y_k_norm;
		float coh = 0.0;
		for (std::size_t k1=0; k1<k && coh<coh_best; ++k1) {
		  float dot = y_k.dot(D0.col(k1));
		  //std::cerr << k1 << ": " << dot << std::endl;
		  coh = std::max(coh, sl::abs(dot));
		}
		//std::cerr << "coh[" << k << " = " << coh << "]" << std::endl;
		if (coh<coh_best) {
		  coh_best = coh;
		  m_k_best = m_k;
		}
		// If high coherence, remove from possible future candidates
		if (coh > 0.9) {
		  std::swap(permutation[i], permutation[permutation.size()-1]);
		  permutation.pop_back();
		} else {
		  // Next
		  ++i;
		}
	      }
	    }

	    // Assign to dictionary
	  
	    // First, try one with lowest coherence...	  
	    if (coh_best<1.0) {
	      d_k = zero_mean(Y[m_k_best]);
	      Eigen::VectorXf d_k_coarse, d_k_fine;
	      coarse_fine_in(d_k_coarse, d_k_fine, dim, Ndim, d_k);
	      if (k<K/8) { d_k=d_k_coarse; } else { d_k=d_k_fine; }
	      d_k_norm = d_k.norm();
	      std::cerr << "Setting max separated: " << m_k_best << " " << coh_best << std::endl;
	    }
	  
	    // ... if not, try picking one vector from examples
	    while (!permutation.empty() && d_k_norm<eps) {
	      std::size_t m_k = permutation.back();
	      d_k = zero_mean(Y[m_k]);
	      d_k_norm = d_k.norm();
	      permutation.pop_back();
	      std::cerr << "Setting random example: " << m_k << std::endl;
	    }
	  
	    // .. if not, pick random non-null vector
	    while (d_k_norm<eps) {
	      std::cerr << "Setting pure random: " << std::endl;
	      d_k.resize(N); d_k.setRandom();
	      d_k = zero_mean(d_k);
	      d_k_norm = d_k.norm();
	    }
	  } // if k

	  // Append d_k to dictionary
	  D0.col(k) = (d_k/d_k_norm);
	} // for k
	std::cerr << std::endl;

	// Generate output
	std::cerr << "COH = " << coherence(D0) << std::endl;

      }
    }

    void dictionary_coder::dictionary_init_in(Eigen::MatrixXf& D0,
					      const std::vector<Eigen::VectorXf>& Y,
					      std::size_t dim,
					      std::size_t Ndim,
					      std::size_t K,
					      float max_coherence) {
      // Default
      SL_USEVAR(dim); SL_USEVAR(Ndim); SL_USEVAR(Y);

      dictionary_init_separated_samples_in(D0,Y,K);
      dictionary_decorrelate_in(D0, max_coherence);
    }

    void dictionary_coder::dictionary_decorrelate_in(Eigen::MatrixXf& D,
						     float max_coherence) {
      // Apply decorrelation only if we have a sensible threshold
      max_coherence = std::abs(max_coherence);
      if (max_coherence > 0.999f) return;
      
      // INK-SVD approach

      // Convert coherence threshold to rotation angle from mean direction
      float theta = 0.5f*std::acos(max_coherence);
      float cos_theta = std::cos(theta);
      float sin_theta = std::sin(theta);

      //const std::size_t N = D.rows();
      const std::size_t K = D.cols();
      const std::size_t max_iter = 100;
      std::size_t decorrelated_count = 1;
      
      for (std::size_t i=0; i<max_iter && decorrelated_count>0; ++i) {
	// Extract all pairs of columns with coherence > max_coherence
	std::vector<std::pair<float, std::pair<std::size_t, std::size_t> > > corr_ij;
	for (std::size_t k1=0; k1<K; ++k1) {
	  for (std::size_t k2=k1+1; k2<K; ++k2) {
	    const float eps=1e-3f;
	    float cor = sl::abs(D.col(k1).dot(D.col(k2)));
	    if (cor>max_coherence+eps) {
	      corr_ij.push_back(std::make_pair(-cor, std::make_pair(k1,k2)));
	    }
	  }
	}
	// Sort by coherence
	std::sort(corr_ij.begin(), corr_ij.end());

	// Decorrelate pairs!
	decorrelated_count = 0;
	std::vector<bool> used_col(K, false);
	for (std::size_t i=0; i<corr_ij.size(); ++i) {
	  //float cor = -corr_ij[i].first;
	  std::size_t k1 = corr_ij[i].second.first;
	  std::size_t k2 = corr_ij[i].second.second;
	  if (!used_col[k1] && !used_col[k2]) {
	    // Decorrelate k2,k2 by moving it far from average
	    Eigen::VectorXf d1 = D.col(k1);
	    Eigen::VectorXf d2 = D.col(k2);
	    float dot = d1.dot(d2);
	    if (sl::abs(dot)>0.9999f) {
	      // Same vectors -- do nothing here hoping it will be fixed by a replacement...
	      std::cerr << "Found duplicated vector: " << k1 << " " << k2 << std::endl;
	    } else if (dot>0.0f) {
	      Eigen::VectorXf v1 = d1+d2; v1.normalize();
	      Eigen::VectorXf v2 = d1-d2; v2.normalize(); 

	      d1 = cos_theta * v1 + sin_theta * v2;
	      d2 = cos_theta * v1 - sin_theta * v2;
	    } else {
	      Eigen::VectorXf v1 = d1-d2; v1.normalize();
	      Eigen::VectorXf v2 = d1+d2; v2.normalize(); 

	      d1 =  cos_theta * v1 + sin_theta * v2;
	      d2 = -cos_theta * v1 + sin_theta * v2;
	    }	      
	    if (k1!= 0) D.col(k1) = d1;
	    if (k2!= 0) D.col(k2) = d2;
	    if (k1==0) std::cerr << "Warning almost const column: " << k2 << std::endl;
	    if (k2==0) std::cerr << "Warning almost const column: " << k1 << std::endl;
	    
	    used_col[k1] = true;
	    used_col[k2] = true;
	    ++decorrelated_count;
	  }
	} // for each pair needing decorrelation
	std::cerr << "Decorrelated " << decorrelated_count << "/" << K << " cols" << std::endl;
      } // for each decorrelation iter
    }

  } // namespace tdm_sparse_coding
} // namespace vic
