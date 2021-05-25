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
#include <vic/tdm_sparse_coding/dictionary_coder_ksvd.hpp>
#include <vic/tdm_sparse_coding/matching_pursuit.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sl/math.hpp>
#include <sl/random.hpp>
#include <sl/clock.hpp>
#include <cassert>
#include <set>
#include <omp.h>

namespace vic {

  // ----------------------------------------------------------------------
  // PCA-L1
  // Kwak, Nojun. "Principal component analysis based on L1-norm maximization."
  // IEEE transactions on pattern analysis and machine intelligence 30.9 (2008): 1672-1680.
  // ----------------------------------------------------------------------

  Eigen::VectorXf pca_l1(Eigen::MatrixXf& E, const Eigen::VectorXf& U0) {
    Eigen::VectorXf U_t = U0;
    
    const std::size_t max_it = 100;
    const float tol = 1e-4f;
    
    bool converged = false;
    for (std::size_t iteration=0;
	 (iteration<max_it) && (!converged);
	 ++iteration) {
      std::size_t zero_count = 0;
      Eigen::VectorXf U_t1 = Eigen::VectorXf::Zero(E.rows());
      for (Eigen::MatrixXf::Index k=0; k<E.cols(); ++k) {
	float dot = U_t.dot(E.col(k));
	if (dot<0) {
	  U_t1 -= E.col(k); 
	} else if (dot>0) {
	  U_t1 += E.col(k);
	} else {
	  ++zero_count;
	}
      }
      if (zero_count && (iteration+1<max_it)) {
	U_t1 += 0.01f*Eigen::VectorXf::Random(U_t1.rows());
      }
      U_t1.normalize();

      float delta = (U_t-U_t1).squaredNorm();
      converged = (delta<tol);
      U_t = U_t1;
    }

    return U_t;
  }

  Eigen::VectorXf pca_l1(Eigen::MatrixXf& E) {
    
    Eigen::MatrixXf::Index   max_index;
    float max_norm = E.colwise().squaredNorm().maxCoeff(&max_index);
    Eigen::VectorXf U0 = E.col(max_index)/std::sqrt(max_norm);

    return pca_l1(E, U0);
  }

  // ----------------------------------------------------------------------
  // K-SVD
  // ----------------------------------------------------------------------

  namespace tdm_sparse_coding {
    
    // Find indices of signals Y and values of gamma whose representation uses atom jj
    void dictionary_coder_ksvd::compressed_rows_in(std::vector<std::vector<std::size_t> >& I_x,
						   std::vector<std::vector<float> >& gamma_j,
						   std::vector<std::vector<std::size_t> >& gamma_j_offset,
						   const std::size_t N,
						   const std::size_t M,
						   const std::size_t column_to_skip,
						   const std::vector<std::size_t>& gamma_offset,
						   const std::vector<std::size_t>& gamma_sz,
						   const std::vector<float>&       gamma_val,
						   const std::vector<std::size_t>& gamma_idx) {
      I_x.clear(); I_x.resize(N);
      gamma_j.clear(); gamma_j.resize(N);
      gamma_j_offset.clear(); gamma_j_offset.resize(N);
      for (std::size_t k_x=0; k_x<M; ++k_x) { // For each input
	const std::size_t gamma_offset_k = gamma_offset[k_x];
	const std::size_t gamma_sz_k = gamma_sz[k_x];
	for (std::size_t kk=0; kk<gamma_sz_k; ++kk) { // For each sparse rep
	  std::size_t jj = gamma_idx[gamma_offset_k+kk];
	  if ((jj != column_to_skip) && (gamma_val[gamma_offset_k+kk] != 0.0)) {
	    I_x[jj].push_back(k_x);
	    gamma_j[jj].push_back(gamma_val[gamma_offset_k+kk]);
	    gamma_j_offset[jj].push_back(gamma_offset_k+kk);
	  }
	}
      }
    }

    void dictionary_coder_ksvd::unused_atom_in(Eigen::VectorXf& atom,
					       std::vector<bool>& is_atom_replaced,
					       std::vector<bool>& is_signal_used,
					       std::size_t jj,
					       const Eigen::MatrixXf& D,
					       const std::vector<float>& sqrt_W,
					       const std::vector<Eigen::VectorXf>& Y,
					       const std::vector<std::size_t>& gamma_offset,
					       const std::vector<std::size_t>& gamma_sz,
					       const std::vector<float>&       gamma_val,
					       const std::vector<std::size_t>& gamma_idx) {
      const std::size_t M=Y.size();

      // No signal uses current atom. It is dead!
      // Replace signal with unused signal with largest error in
      // current dictionary
      std::vector<std::size_t> unused_permutation;
      for (std::size_t k=0; k<M; ++k) {
	if (!is_signal_used[k]) {
	  unused_permutation.push_back(k);
	}
      }
      const std::size_t Max_signals = 5000;
      if (unused_permutation.size() > Max_signals) {
	std::random_shuffle(unused_permutation.begin(), unused_permutation.end());
	unused_permutation.resize(Max_signals);
      }
      // Choose signal with largest error in current dictionary
      std::size_t k_best = std::size_t(-1);
      float       e2_best = -1.0f;
      for (std::size_t k=0; k<unused_permutation.size(); ++k) {
	std::size_t k_x = unused_permutation[k];
	Eigen::VectorXf d_x = sqrt_W[k_x] * Y[k_x];
	float d2 = d_x.squaredNorm();
	if (d2 > 1e-10f) {
	  const std::size_t gamma_offset_k = gamma_offset[k_x];
	  const std::size_t gamma_sz_k = gamma_sz[k_x];
	  for (std::size_t kk=0; kk<gamma_sz_k; ++kk) {
	    d_x -= gamma_val[gamma_offset_k+kk] * D.col(gamma_idx[gamma_offset_k+kk]);
	  }
	  float e2_x = d_x.squaredNorm();
	  if (e2_x>e2_best) {
	    k_best = k_x;
	    e2_best = e2_x;
	  }
	}
      }
      float atom_norm = 0.0f;
      if (k_best != std::size_t(-1)) {
	atom = zero_mean(Y[k_best]);
	atom_norm = atom.norm();
	if (atom_norm >=1.0e-7f) {
	  is_signal_used[k_best] = true;
	}
      } 
      // Example signals not sufficient ????
      while (atom_norm < 1.0e-7f) {
	atom.resize(D.rows()); atom.setRandom();
	atom = zero_mean(atom);
	atom_norm = atom.norm();
      }
      atom /= atom_norm;
      
      is_atom_replaced[jj] = true;
    }
    
    void dictionary_coder_ksvd::improve_atom_l2_in(Eigen::VectorXf& atom,
						   const std::vector<float>& gamma_j,
						   const std::vector<std::size_t>& I_x,
						   std::size_t jj,
						   const Eigen::MatrixXf& D,
						   const std::vector<float>& sqrt_W,
						   const std::vector<Eigen::VectorXf>& Y,
						   const std::vector<std::size_t>& gamma_offset,
						   const std::vector<std::size_t>& gamma_sz,
						   const std::vector<float>&       gamma_val,
						   const std::vector<std::size_t>& gamma_idx) {
      // atom = Y_I*gamma_j' - D*(Gamma_I*gamma_j') + D_j*(gamma_j*gamma_j');
      // atom = atom/norm(atom);

      atom.setZero(D.rows());
      float gamma_j_dot_gamma_j = 0.0f;
      Eigen::VectorXf gamma_prime; gamma_prime.setZero(D.cols()); 
      const std::size_t I_x_size = I_x.size();
      for (std::size_t k_x=0; k_x< I_x_size; ++k_x) {
	const std::size_t ii_k = I_x[k_x];
	atom += gamma_j[k_x]*sqrt_W[ii_k]*Y[ii_k];
	gamma_j_dot_gamma_j += gamma_j[k_x] * gamma_j[k_x];
	
	const std::size_t gamma_offset_k = gamma_offset[ii_k];
	const std::size_t gamma_sz_k = gamma_sz[ii_k];
	for (std::size_t kk=0; kk<gamma_sz_k; ++kk) {
	  gamma_prime[gamma_idx[gamma_offset_k+kk]] += gamma_j[k_x] * gamma_val[gamma_offset_k+kk];
	}
      }
      
      atom -= D * gamma_prime;
      atom += D.col(jj) * gamma_j_dot_gamma_j; // remove self reference
      atom /= atom.norm(); // FIXME check zero?
    }

    
    void dictionary_coder_ksvd::improve_atom_l1_in(Eigen::VectorXf& atom,
						   const std::vector<float>& gamma_j,
						   const std::vector<std::size_t>& I_x,
						   std::size_t jj,
						   const Eigen::MatrixXf& D,
						   const std::vector<float>& sqrt_W,
						   const std::vector<Eigen::VectorXf>& Y,
						   const std::vector<std::size_t>& gamma_offset,
						   const std::vector<std::size_t>& gamma_sz,
						   const std::vector<float>&       gamma_val,
						   const std::vector<std::size_t>& gamma_idx) {
      // Compute matrix of residuals when removing column jj from dictionary
      const std::size_t I_x_size = I_x.size();
      Eigen::MatrixXf E_sub(D.rows(),I_x_size);

#pragma omp parallel for
      for (std::size_t i=0; i<I_x_size; ++i) {
	const std::size_t ii_k = I_x[i];

	const std::size_t gamma_offset_k = gamma_offset[ii_k];
	const std::size_t gamma_sz_k = gamma_sz[ii_k];

	E_sub.col(i) = sqrt_W[ii_k] * Y[ii_k];

	for (std::size_t kk=0; kk<gamma_sz_k; ++kk) {
	  std::size_t jj_kk = gamma_idx[gamma_offset_k+kk];
	  if (jj_kk != jj) {
	    E_sub.col(i) -= gamma_val[gamma_offset_k+kk] * D.col(jj_kk);
	  }
	}
      }

      // Optimize column jj to reduce l1 norm of residual
#if 0
      atom = pca_l1(E_sub);
#else
      improve_atom_l2_in(atom,
			 gamma_j, I_x, jj, D, sqrt_W, Y,
			 gamma_offset, gamma_sz, gamma_val, gamma_idx);
      atom = pca_l1(E_sub, atom);
#endif
    }

    
    void dictionary_coder_ksvd::improve_gamma_in(std::vector<float>& gamma_j,
						 const Eigen::VectorXf& atom,
						 const std::vector<std::size_t>& I_x,
						 std::size_t jj,
						 const Eigen::MatrixXf& D,
						 const std::vector<float>& sqrt_W,
						 const std::vector<Eigen::VectorXf>& Y,
						 const std::vector<std::size_t>& gamma_offset,
						 const std::vector<std::size_t>& gamma_sz,
						 const std::vector<float>&       gamma_val,
						 const std::vector<std::size_t>& gamma_idx) {
      // gamma_j = atom'*Y_I - (atom'*D)*Gamma_I + (atom'*Dj)*gamma_j;
      const std::size_t I_x_size = I_x.size();
      std::vector<float> gamma_j_prime(I_x_size, 0.0f);
      float atom_dot_Dj = atom.dot(D.col(jj));
      Eigen::RowVectorXf Dt_atom = atom.transpose() * D;
      //#pragma omp parallel for -- FIXME removed -- slow
      for (std::size_t k_x=0; k_x< I_x_size; ++k_x) {
	const std::size_t ii_k = I_x[k_x];
	gamma_j_prime[k_x] = sqrt_W[ii_k]*atom.dot(Y[ii_k]);
	
	const std::size_t gamma_offset_k = gamma_offset[ii_k];
	const std::size_t gamma_sz_k = gamma_sz[ii_k];
	for (std::size_t kk=0; kk<gamma_sz_k; ++kk) {
	  gamma_j_prime[k_x] -= Dt_atom[gamma_idx[gamma_offset_k+kk]] * gamma_val[gamma_offset_k+kk]; // FIXME 
	}
	
	gamma_j_prime[k_x] += atom_dot_Dj*gamma_j[k_x]; // Remove self reference
      }
      for (std::size_t k_x=0; k_x< I_x_size; ++k_x) {
	gamma_j[k_x] = gamma_j_prime[k_x];
      }
    }
    
    void dictionary_coder_ksvd::dictionary_cleanup_in(Eigen::MatrixXf& D,
						      std::vector<bool>& is_signal_used,
						      const std::vector<bool>& is_atom_replaced,
						      const std::vector<float>& sqrt_W,
						      const std::vector<Eigen::VectorXf>& Y,
						      const std::vector<std::size_t>& gamma_offset,
						      const std::vector<std::size_t>& gamma_sz,
						      const std::vector<float>&       gamma_val,
						      const std::vector<std::size_t>& gamma_idx,
						      float mutual_incoherence_threshold,
						      std::size_t use_threshold) {
      const std::size_t N=D.cols();
      const std::size_t M=Y.size();

      std::size_t replaced_unused_count = 0;
      std::size_t replaced_coherent_count = 0;
      
      std::vector<std::size_t> use_count(N,0);
      std::vector<float> err2(M,0.0f);
      for (std::size_t k_x=0; k_x<M; ++k_x) {
	Eigen::VectorXf d_x = sqrt_W[k_x]*Y[k_x];
	float d2 = d_x.squaredNorm();
	if (d2 > 1e-10f) {
	  const std::size_t gamma_offset_k = gamma_offset[k_x];
	  const std::size_t gamma_sz_k = gamma_sz[k_x];
	  for (std::size_t kk=0; kk<gamma_sz_k; ++kk) {
	    std::size_t gi = gamma_idx[gamma_offset_k+kk];
	    float gv = gamma_val[gamma_offset_k+kk];
	    if (gv != 0.0) {
	      d_x -= gv * D.col(gi);
	      if (sl::abs(gv)>1e-10f) ++use_count[gi];
	    }
	  }
	  err2[k_x] = d_x.squaredNorm();
	}
      }
#if 0
      std::cerr << "Dict col use count: ";
      for (std::size_t j=0; j<std::size_t(N); ++j) {
	std::cerr << " ";
	if (use_count[j] < use_threshold) {
	  std::cerr << "[" << use_count[j] << "]";
	} else {
	  std::cerr << use_count[j];
	}
      }
      std::cerr << std::endl;
#endif
      
      const float tol2 = mutual_incoherence_threshold*mutual_incoherence_threshold;
      for (std::size_t j=0; j<std::size_t(N); ++j) {
	if ((j==0) || is_atom_replaced[j]) {
	  // Do not optimize
	} else {
	  Eigen::VectorXf Gj = D.transpose()*D.col(j);
	  Gj(j) = 0.0f;
	  float G2max = 0.0f;
	  for (std::size_t k=0; k<std::size_t(Gj.rows()); ++k) {
	    G2max = std::max(G2max, Gj(k)*Gj(k));
	  }
	  if (G2max>tol2 || (use_count[j]<use_threshold)) {
	    float       e2_best = -1.0f;
	    std::size_t k_best = std::size_t(-1);
	    for (std::size_t k_x=0; k_x<M; ++k_x) {
	      if (!is_signal_used[k_x]) {
		if (err2[k_x]>e2_best) {
		  e2_best = err2[k_x];
		  k_best = k_x;
		}
	      }
	    }
	    float           atom_norm=0.0f;
	    Eigen::VectorXf atom;
	    if (e2_best>=0.0f) {
	      atom = zero_mean(Y[k_best]);
	      atom_norm = atom.norm();
	      is_signal_used[k_best] = true;
	    }
	    while (atom_norm < 1.0e-6f) {
	      atom.resize(D.rows()); atom.setRandom();
	      atom_norm = atom.norm();
	    }
	    atom /= atom_norm;
	    D.col(j) = atom;

	    if (use_count[j]<use_threshold) {
	      ++ replaced_unused_count;
	      std::cerr << "Replaced column " << j << " (used only " << use_count[j] << " times)" << std::endl;
	    } else {
	      ++ replaced_coherent_count;
	      std::cerr << "Replaced column " << j << " (coherent)" << std::endl;
	    }
	  } // if atom must be replaced
	}
      } // for each atom

      if (replaced_coherent_count) std::cerr << "     Dictionary cleanup: replaced " << replaced_coherent_count << "/" << N << " too coherent columns" << std::endl;
      if (replaced_unused_count) std::cerr << "     Dictionary cleanup: replaced " << replaced_unused_count << "/" << N << " unused columns" << std::endl; 
    } // dictionary_coder_ksvd::dictionary_cleanup_in
  
    void dictionary_coder_ksvd::refine_in(Eigen::MatrixXf& D,
					  error_evaluator& refine_stats,
					  const  std::vector<float>& W,
					  const  std::vector<Eigen::VectorXf>& Y,
					  std::size_t K,
					  float tol,
					  float max_coh,
					  std::size_t dictionary_update_cycle_count,
					  bool is_l1_pca_enabled,
					  std::size_t epoch_count) {
      SL_TRACE_OUT(1) << "W.sz " << W.size() << ", Y.sz " << Y.size() << ", K " << K << ", tol " << tol << std::endl;
      double best_rmse = 1e30;

      const std::size_t N=D.cols();
      const std::size_t M=Y.size();

      std::vector<float> sqrt_W(M);
      for (std::size_t r=0; r<M; ++r) {
	sqrt_W[r] = std::sqrt(W[r]);
      }

      Eigen::MatrixXf best_D = D;
      std::size_t     best_epoch = std::size_t(-1);
      //      double          best_rmse = 1e30f;
      
      sl::real_time_clock ck;
      ck.restart();
      for (std::size_t epoch=0; epoch<epoch_count+1; ++epoch) {
	// -- Sparse encode Y given current dictionary
	std::vector<std::size_t>  gamma_offset;
	std::vector<std::size_t>  gamma_sz;
	std::vector<float>        gamma_val;
	std::vector<std::size_t>  gamma_idx;

	const Eigen::MatrixXf DtD = D.transpose()*D;
	omp_batch_in(gamma_offset, gamma_sz, gamma_val, gamma_idx,
		     D, DtD, Y, 
		     K, tol);

	double mean_atom_count= 0.0;
	for (std::size_t r=0; r<M; ++r) {
	  mean_atom_count += gamma_sz[r];
	}
	mean_atom_count/=double(M);
	
	// -- Update learning stats
	std::size_t N_threads = omp_get_num_procs();
	std::vector<error_evaluator> stats(N_threads);
	for (std::size_t tid=0; tid<N_threads; ++tid) {
	  stats[tid].set_is_verbose(false);
	}
#pragma omp parallel for
	for (std::size_t r=0; r<M; ++r) {
	  const std::size_t tid = omp_get_thread_num();

	  const Eigen::VectorXf& y_r = Y[r];
	  const float            sqrt_w_r = sqrt_W[r];
	  Eigen::VectorXf  y_tilde(y_r.size());
	  y_tilde.setConstant(0.0f);
	  const std::size_t gamma_offset_r = gamma_offset[r];
	  const std::size_t gamma_sz_r = gamma_sz[r];
	  for (std::size_t kk=0; kk<gamma_sz_r; ++kk) { // For each sparse rep
	    y_tilde += gamma_val[gamma_offset_r+kk] * D.col(gamma_idx[gamma_offset_r+kk]);
	  }
	  stats[tid].put(y_r,y_tilde,sqrt_w_r*sqrt_w_r);	  
	}
	for (std::size_t tid=1; tid<N_threads; ++tid) {
	  stats[0].accumulate(stats[tid]);
	}
	refine_stats = stats[0];
	
	// Apply coreset weighting to gammas
	for (std::size_t i=0; i<M; ++i) {
	  float sqrt_w_i = sqrt_W[i];
	  const std::size_t gamma_offset_i = gamma_offset[i];
	  const std::size_t gamma_sz_i = gamma_sz[i];
	  for (std::size_t kk=0; kk<gamma_sz_i; ++kk) { 
	    gamma_val[gamma_offset_i+kk] *= sqrt_w_i;
	  }
	}
	       
	std::cerr << "[ " << epoch << "]: ";
	if ((epoch == 0) || (stats[0].last_rmse()<best_rmse)) {
	  std::cerr << "(*)";
	  best_D = D;
	  best_rmse = stats[0].last_rmse();
	  best_epoch = epoch;
	} else {
	  std::cerr << "( )";
	}
	
	std::cerr <<
	  " RMSE = " << stats[0].last_rmse() <<
	  " NRMSE = " << stats[0].last_nrmse() <<
	  " PSNR = " << stats[0].last_PSNR_dB() << " dB" <<
	  " MAXE = " << stats[0].last_emax() << 
	  " (y: " << stats[0].last_ymin() << " ... " << stats[0].last_ymax() << ") " << 
	  " k_avg = " << mean_atom_count <<
	  " COH = " << coherence(D) << 
          " elaps = " << sl::human_readable_duration(ck.elapsed()) << std::endl;

	if (!(epoch+1<epoch_count+1)) {
	  std::cerr << "[ " << epoch << "]: ";
	  if (best_epoch != epoch) {
	    std::cerr << "Reverting to best epoch: " << best_epoch;
	    D = best_D;
	  } else {
	    std::cerr << "Done." << std::endl;
	  }
	  std::cerr << std::endl;
	} else {
	  // -- Update dictionary atoms given the current sparse
	  // -- representation gamma.
	  
	  std::vector<bool> is_atom_replaced(N, false); // Mark each atom replaced
	  std::vector<bool> is_signal_used(M, false); // signals used to replace "dead" atoms
	  std::vector<std::size_t> permutation(N);
	  for (std::size_t j=0; j<std::size_t(N); ++j) {
	    permutation[j] = j;
	  }
	  std::random_shuffle(permutation.begin(), permutation.end());
	  
	  // Find indices of signals Y whose representation uses atom j
	  std::vector< std::vector<std::size_t> > I_x;
	  std::vector< std::vector<float> > gamma_j;
	  std::vector< std::vector<std::size_t> > gamma_j_offset;
	  compressed_rows_in(I_x, gamma_j, gamma_j_offset,
			     N, M, 0, 
			     gamma_offset, gamma_sz, gamma_val, gamma_idx);

	  // Multiple Dictionary update cycles as in
	  // Smith & Elad, Improving Dictionary Learning: Multiple Dictionary
	  // Updates and Coefficient Reuse, IEEE Signal Processing Letters 20(1), 2013.
	  for (std::size_t duc_i = 0; duc_i < dictionary_update_cycle_count; ++duc_i) {
	    // Updated one atom at a time
	    for (int j=0; j<int(N); ++j) {
	      // std::cerr << "TID" << omp_get_thread_num() << std::endl;
	      std::size_t jj = permutation[j];
	      if (jj==0) {
		// Do not improve this atom
	      } else {
		Eigen::VectorXf atom;
		if (I_x[jj].empty()) {
		  std::cerr << "*********** Replacing unused atom " << jj << std::endl; // FIXME
		  unused_atom_in(atom,
				 is_atom_replaced, is_signal_used,
				 jj,
				 D, sqrt_W, Y,
				 gamma_offset, gamma_sz, gamma_val, gamma_idx);
		} else {
		  if (is_l1_pca_enabled) {
		    improve_atom_l1_in(atom,
				       gamma_j[jj], I_x[jj],
				       jj,
				       D, sqrt_W, Y,
				       gamma_offset, gamma_sz, gamma_val, gamma_idx);
		  } else {
		    improve_atom_l2_in(atom,
				       gamma_j[jj], I_x[jj],
				       jj,
				       D, sqrt_W, Y,
				       gamma_offset, gamma_sz, gamma_val, gamma_idx);
		  }
		  improve_gamma_in(gamma_j[jj],
				   atom, I_x[jj],
				   jj,
				   D, sqrt_W, Y,
				   gamma_offset, gamma_sz, gamma_val, gamma_idx);
		}
		// -- Update current representation
		D.col(jj) = atom;
		for (std::size_t k_x=0; k_x< I_x[jj].size(); ++k_x) {
		  gamma_val[gamma_j_offset[jj][k_x]] = gamma_j[jj][k_x];
		}
	      }
	    } // for each atom
	  } // for each dictionary update cycle
	  
	  if (epoch+1 != epoch_count) {

	    const float mutual_incoherence_threshold = 0.999f; // 0.99f
	    const std::size_t use_threshold = 2; // sl::median(4, 2, int(0.1f*float(Y.size())/float(D.cols())));
	    dictionary_cleanup_in(D,
				  is_signal_used,
				  is_atom_replaced,
				  sqrt_W, Y,
				  gamma_offset, gamma_sz, gamma_val, gamma_idx,
				  mutual_incoherence_threshold,
				  use_threshold);

	    if (max_coh < 0.999f) {
	      dictionary_decorrelate_in(D, max_coh);
	    }
	  }
	} // if not last epoch
      } // for each iteration
    } // dictionary_coder_ksvd::refine_in

    // ========================================================================
    // Class implementation
    // ========================================================================

  dictionary_coder_ksvd::dictionary_coder_ksvd(): max_coherence_(1.0f), dictionary_update_cycle_count_(1), is_l1_pca_enabled_(false) {
    }
      
    dictionary_coder_ksvd::~dictionary_coder_ksvd() {
    }

    dictionary_coder_ksvd* dictionary_coder_ksvd::clone() const {
      return new dictionary_coder_ksvd(*this);
    }
    
    void dictionary_coder_ksvd::reset() {
      init(dim_, Ndim_, K_, S_, T_, max_coherence_, dictionary_update_cycle_count_, is_l1_pca_enabled_);
    }
    
    void  dictionary_coder_ksvd::train_on_coreset(const std::vector<float>&  W,
						  const std::vector<Eigen::VectorXf>& Y) {
      if (is_using_precomputed_init_dictionary()) {
	std::cerr << "===========================================================" << std::endl;
	std::cerr << "KSVD using precomputed init dictionary" << std::endl;
	D_ = D_init_;
      } else {
	std::cerr << "===========================================================" << std::endl;
	std::cerr << "KSVD computing init dictionary from data" << std::endl;
	dictionary_init_in(D_, Y, dim_, Ndim_, K_, max_coherence_);
      }

      invalidate_precomputed_coding_data();
      precompute_coding_data();

      std::cerr << "===========================================================" << std::endl;
      std::cerr << "KSVD One pass training" << std::endl;
      std::cerr << "  Coherence threshold = " << max_coherence_ << std::endl;
      std::cerr << "  Atom update =  " << (is_l1_pca_enabled_ ? "L1" : "L2") << std::endl;
      std::cerr << "  Dictionary update cycles per epoch = " << dictionary_update_cycle_count_ << std::endl;
      std::cerr << "===========================================================" << std::endl;
      
      if (training_epoch_count_ > 0) {
	refine_in(D_, last_stats_, W, Y, S_, T_,
		  max_coherence_, dictionary_update_cycle_count_, is_l1_pca_enabled_,
		  training_epoch_count_);
      }
      
      std::cerr << "===========================================================" << std::endl;

      invalidate_precomputed_coding_data();
      precompute_coding_data();
    }

    void dictionary_coder_ksvd::init(std::size_t dim, std::size_t Ndim,
				     std::size_t K, std::size_t S, float T,
				     float max_coh,
				     std::size_t dictionary_update_cycle_count,
				     bool is_l1_pca_enabled) {
      dim_ = dim;
      Ndim_ = Ndim;
      N_ = sl::ipow(Ndim,dim);
      K_ = K;
      S_ = std::min(S,std::min(K_,N_));
      T_ = T;
      max_coherence_ = max_coh;
      dictionary_update_cycle_count_ = dictionary_update_cycle_count;
      is_l1_pca_enabled_ = is_l1_pca_enabled;
    
      dictionary_init_random_dictionary_in(D_,N_,K_);
            
      invalidate_precomputed_coding_data();
    }

  } // namespace tdm_sparse_coding
} // namespace vic
