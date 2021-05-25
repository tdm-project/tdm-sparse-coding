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
#include <vic/tdm_sparse_coding/matching_pursuit.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <sl/math.hpp>
#include <sl/random.hpp>
#include <cassert>
#include <set>
#include <omp.h>

namespace vic {
  // ----------------------------------------------------------------------
  // Least squares
  // ----------------------------------------------------------------------

  namespace tdm_sparse_coding {
     
    void least_squares_in(float *gamma_val,
			  const std::size_t *gamma_idx,
			  const Eigen::MatrixXf& D,
			  const Eigen::VectorXf& x,
			  std::size_t K) {
      assert(K>0);

      // Double precision version
      Eigen::MatrixXd Di(D.rows(),K);
      for (std::size_t k1=0; k1<K; ++k1) {
	Di.col(k1) = D.col(gamma_idx[k1]).cast<double>();
      }
      Eigen::VectorXd gamma_I = Di.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(x.cast<double>());
      for (std::size_t k1=0; k1<K; ++k1) {
	gamma_val[k1] = float(gamma_I[k1]);
      }
    }

  } // namespace tdm_sparse_coding
  
  // ----------------------------------------------------------------------
  // Orthogonal matching pursuit
  // ----------------------------------------------------------------------

  namespace tdm_sparse_coding {
     
    inline std::size_t iabsdist(std::size_t a, std::size_t b) {
      if (a>b) return a-b; else return b-a;
    }

    inline std::size_t iabsdist(const std::size_t* old_idx,
				const std::size_t N,
				std::size_t new_idx) {
      if (N==0) return new_idx;
      std::size_t result = iabsdist(new_idx, old_idx[0]);
      for (std::size_t i=1; i<N; ++i) {
	std::size_t d_i = iabsdist(new_idx, old_idx[i]);
	if (d_i<result) result=d_i;
      }
      return result;
    }

    // OMP Batch -- optimized fixed sparsity and tolerance-driven version
    void omp_batch_in(float       *gamma_val,
		      std::size_t *gamma_idx,
		      const Eigen::VectorXf& Dtx, // = trans(D)*x
		      const Eigen::MatrixXf& DtD, // = trans(D)*D
		      std::size_t K,
		      float       xtx,            // = trans(x)*x
		      float tol,
		      std::size_t* out_K,
		      float* out_err2,  // Achieved error squared
		      float* out_err2_k) { // size K+1, on output is the error for 0..out_K
      std::size_t Kmax = std::size_t(Dtx.rows());
      assert(K<=Kmax);
	  
      // Init sparse code (null representation)
      for (std::size_t k1= 0; k1<K; ++k1) {
	gamma_idx[k1] = 0;
	gamma_val[k1] = 0.0f;
      }
      if (out_K) (*out_K) = 0;

      // Init error computation
      const double tol2 = double(tol)*double(tol);
      double err2 = xtx;
      double deltaprev = 0.0;

      if (out_err2_k) out_err2_k[0] = float(err2); // Err for sparsity 0
      
      if (tol2>0.0 && err2<tol2) {
	// Tolerance reached with zero vector 
	if (out_err2) (*out_err2) = float(err2);
	if (out_err2_k) {
	  for (std::size_t k=0; k<K+1;++k) {
	    out_err2_k[k] = float(err2);
	  }
	}
	// Done!
	return;
      } 
      
      // Allocate temporaries
      Eigen::VectorXf DtD_Ik(K);
      Eigen::VectorXf Dtx_I(K+1);
      
      // Init Cholesky decomposition
      Eigen::MatrixXf Lchol = Eigen::MatrixXf::Zero(K,K);
      Lchol(0,0) = 1.0f;

      // Init Alpha
      Eigen::VectorXf alpha = Dtx;

      const bool is_error_computation_enabled = (out_err2) || (out_err2_k) || (tol>0.0f);

      for (std::size_t k=0; k<K; ++k) {
	//////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////
	/////// FIXME FIXME FIXME - TEST LIMIT BITS 
	/////////////////////////////////////////////////////////////////////

	// Find most aligned basis vec
	std::size_t k_star = 0;
	float       a_star = sl::abs(alpha[0]);
	for (std::size_t k1=1; k1<Kmax; ++k1) {
	  const float a = sl::abs(alpha[k1]);
	  if (a>a_star) {
	    a_star = a; k_star = k1;
	  }
	}

	// Compute first approximation
	if (k==0) {
	  // First value -- just dot product
	  gamma_idx[0] = k_star;
	  gamma_val[0] = Dtx[k_star];
	  if (out_K) (*out_K) = 1;
	}

	// Exit if almost zero
	if (a_star<1.0e-6f) { /*Abort*/ break; } 

	// Compute next approximation
	if (k>0) {
	  // Other values -- must recompute least squares solution using all basis vecs

	  // Incremental Cholesky decomposition: compute next row of Lchol
	  for (std::size_t k1=0; k1<k; ++k1) {
	    DtD_Ik(k1) = DtD(gamma_idx[k1],k_star);
	  }
	  Eigen::VectorXf w = Lchol.topLeftCorner(k, k).triangularView<Eigen::Lower>().solve(DtD_Ik.topLeftCorner(k,1));
	  const float one_minus_w2 = 1.0f-w.squaredNorm();
	  if (one_minus_w2 <= 1e-14f) { /*Abort*/ break; }
	  Lchol.block(k,0,1,k) = w.transpose();
	  Lchol(k,k) = std::sqrt(one_minus_w2);

	  gamma_idx[k] = k_star;
	  
	  // Compute gamma
	  for (std::size_t k1=0; k1<=k; ++k1) {
	    Dtx_I(k1) = Dtx(gamma_idx[k1]);
	  }
	  Lchol.topLeftCorner(k+1, k+1).triangularView<Eigen::Lower>().solveInPlace(Dtx_I.topLeftCorner(k+1,1));
	  Lchol.topLeftCorner(k+1, k+1).transpose().triangularView<Eigen::Upper>().solveInPlace(Dtx_I.topLeftCorner(k+1,1));

	  // Push next selected vector and update projection
	  for (std::size_t k1=0; k1<=k; ++k1) {
	    gamma_val[k1] = Dtx_I(k1);
	  }
	  if (out_K) (*out_K) = k+1;
	} // if not first value

       
	if ((k+1!=K) || is_error_computation_enabled) {
	  // Compute residual similarity with basis vecs
	  // r = x-Dg
	  // similarity = max trans(d_k) * r
	  // trans(D)r = trans(D)*x - trans(D)*D.g = Dtx - DtD * g  
	  alpha = Dtx;
	  for (std::size_t k1=0; k1<=k; ++k1) {
	    alpha -= DtD.col(gamma_idx[k1]) * gamma_val[k1];
	  }
	}
	
	// Compute error if needed
	if (is_error_computation_enabled) {
	  // From Rubinstein et al, "Efficient implementation of the K-SVD
	  // algorithm using batch orthogonal matching pursuit":
	  //   beta_I = DtD_I * gamma_I
	  //   alpha = Dtx - beta_I
	  //   delta = gamma_I^T * beta_I
	  //   err2 = err2 - delta + delta_prev
	  //   delta_prev = delta
	  // My version without vector temporaries:
	  //    alpha = Dtx - DtD_I * gamma_I
	  //    delta = gamma_I^T * (Dtx - alpha) 
	  //    err2 = err2 - delta + delta_prev
	  //    delta_prev = delta
	  
	  double delta = 0.0;
	  for (std::size_t k1=0; k1<=k; ++k1) {
	    delta += double(gamma_val[k1]) * (double(Dtx[gamma_idx[k1]]) - double(alpha[gamma_idx[k1]]));
	  }
	  err2 = std::max(0.0, err2 - delta + deltaprev);
	  deltaprev=delta;
	  if (out_err2) (*out_err2) = float(err2);
	  if (out_err2_k) {
	    out_err2_k[k+1] = float(err2);
	    for (std::size_t kk=k+1; kk<K;++kk) {
	      out_err2_k[kk+1] = out_err2_k[k+1];
	    }
	  }

	  if (tol2>0.0 && err2<tol2) { /*Tolerance reached - exit*/ break; }
	}
      } // for each k

      if (!out_err2_k) {
      	// We have no history -- we can thus cleanup the solution
	// without preserving greedy ordering
	
	// Cleanup: move all null gamma values to end of array
	std::size_t k_nonzero = 0;
	for (std::size_t k = 0; k <K; ++k) {
          if (sl::abs(gamma_val[k])>1e-10f) {
	    gamma_idx[k_nonzero] = gamma_idx[k];
	    gamma_val[k_nonzero] = gamma_val[k];
	    ++k_nonzero;
	  }
	}
	// Communicate number of nonzeros and clear rest of array
	if (out_K) (*out_K) = k_nonzero;
	for (std::size_t k=k_nonzero; k < K; ++k) {
	  gamma_idx[k] = 0;
	  gamma_val[k] = 0.0f;
	}
      }
    }
    
    void omp_batch_in(std::vector<std::size_t>&  gamma_offset,
		      std::vector<std::size_t>&  gamma_sz,
		      std::vector<float>&        gamma_val,
		      std::vector<std::size_t>&  gamma_idx,
		      const Eigen::MatrixXf& D,
		      const Eigen::MatrixXf& DtD,
		      const std::vector<Eigen::VectorXf>& X,
		      std::size_t K,
		      float tol) {
      const std::size_t X_count = X.size();
      //const std::size_t X_dim = D.rows();
      		     
      gamma_offset.clear();
      gamma_sz.clear();
      gamma_val.clear();
      gamma_idx.clear();

      const Eigen::MatrixXf Dt = D.transpose();

      const std::size_t BATCH_SIZE=8192;
      std::vector<std::size_t>  batch_gamma_sz(BATCH_SIZE);
      std::vector<float>        batch_gamma_val(BATCH_SIZE*K);
      std::vector<std::size_t>  batch_gamma_idx(BATCH_SIZE*K);
      for (std::size_t i_bgn=0; i_bgn<X_count; i_bgn+=BATCH_SIZE) {
	std::size_t i_end = std::min(X_count, i_bgn+BATCH_SIZE);
	
#pragma omp parallel for schedule(static)
	for (std::size_t i=i_bgn; i<i_end; ++i) {
	  const Eigen::VectorXf& x_i = X[i];

	  const double x2_i = x_i.squaredNorm();
	  const Eigen::VectorXf Dtx_i = Dt*x_i;
	  std::size_t K_out = 0;
	  omp_batch_in(&(batch_gamma_val[(i-i_bgn)*K]),
		       &(batch_gamma_idx[(i-i_bgn)*K]),
		       Dtx_i,
		       DtD,
		       K,
		       float(x2_i),
		       tol,
		       &K_out);
	  // Keep fixed layout for zero-tolerance, otherwise compress
	  batch_gamma_sz[i-i_bgn] = (tol==0.0f) ? K : K_out;
	  for (std::size_t k=K_out; k<batch_gamma_sz[i-i_bgn]; ++k) {
	    batch_gamma_val[(i-i_bgn)*K+k] = 0.0f;
	    batch_gamma_idx[(i-i_bgn)*K+k] = 0;
	  }
	}
	
	// Preallocate output data after first batch
	if (i_bgn==0) {
	  gamma_sz.reserve(X_count);
	  gamma_offset.reserve(X_count);
	  std::size_t gamma_count = X_count*K;
	  if (tol!=0.0f) {
	    // Variable rate -- Estimate size from first batch
	    std::size_t batch_gamma_count = 0;
	    for (std::size_t i=i_bgn; i<i_end; ++i) {
	      batch_gamma_count += batch_gamma_sz[(i-i_bgn)];
	    }
	    gamma_count = std::size_t(1.2f*float(X_count)*float(batch_gamma_count)/float(i_end-i_bgn));
	  }
	  // Allocate
	  gamma_val.reserve(gamma_count);
	  gamma_idx.reserve(gamma_count);
	}
	// Merge batch into output
	for (std::size_t i=i_bgn; i<i_end; ++i) {
	  const std::size_t k_i = batch_gamma_sz[i-i_bgn];
	  gamma_sz.push_back(k_i);
	  gamma_offset.push_back(gamma_val.size());
	  for (std::size_t k=0; k<k_i; ++k) {
	    gamma_val.push_back(batch_gamma_val[(i-i_bgn)*K+k]);
	    gamma_idx.push_back(batch_gamma_idx[(i-i_bgn)*K+k]);
	  }
	}
      } // For each batch
    }
    
  } // namespace tdm_sparse_coding


  
  // ----------------------------------------------------------------------
  // Helper
  // ----------------------------------------------------------------------
  namespace tdm_sparse_coding {

    void sequential_least_squares_in(float *gamma_val,
				     std::size_t *gamma_idx,
				     const Eigen::MatrixXf& D,
				     const Eigen::VectorXf& x,
				     std::size_t K) {
      assert(K>0);

      gamma_val[0] = x.dot(D.col(gamma_idx[0]));
      Eigen::VectorXf dx = x-gamma_val[0]*D.col(gamma_idx[0]);
      float best_e2 = dx.squaredNorm();
      //std::size_t best_k = 0;
      for (std::size_t k=1; k<K; ++k) {
	 Eigen::MatrixXf Di(D.rows(),k+1);
	 for (std::size_t k1=0; k1<=k; ++k1) {
	   Di.col(k1) = D.col(gamma_idx[k1]);
	 }
	 Eigen::VectorXf gamma_I = Di.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(x);
	 Eigen::VectorXf dx = x;
	 for (std::size_t k1=0; k1<=k; ++k1) {
	   dx -= gamma_I[k1]*D.col(gamma_idx[k1]);
	 }
	 float e2 = dx.squaredNorm();
	 if (e2<best_e2) {
	   best_e2=e2; //best_k=k;
	   for (std::size_t k1=0; k1<=k; ++k1) {
	     gamma_val[k1] = gamma_I[k1];
	   }
	 } else {
	   // We do not improve error - stop here and clear rest of gammas
	   for (std::size_t k1=k; k1<K; ++k1) {
	     gamma_val[k1]=0.0f;
	     gamma_idx[k1]=gamma_idx[0];
	   }
	   k=K; break;
	 }
      }
    }
  }

  // ----------------------------------------------------------------------
  // Order recursive matching pursuit
  // ----------------------------------------------------------------------

  namespace tdm_sparse_coding {
    
    // Implementation based on QR factorization
    // Karl Skretting and John Hakon Husoy,
    // Partial search vector selection for sparse signal representation
    // NORSIG'03
    // 
    void ormp_in(float *gamma_val,
		 std::size_t *gamma_idx,
		 const Eigen::MatrixXf& D,
		 const Eigen::VectorXf& x,
		 std::size_t K) {
      const std::size_t N = std::size_t(D.cols());
      assert(K<=N);
      assert(K>0);

      Eigen::MatrixXf R(K, N); R.setZero(); // R matrix of QR factorization
      Eigen::VectorXf e(N); e.fill(1.0f);
      Eigen::VectorXf u(N); u.fill(1.0f);
      Eigen::VectorXf c = D.transpose() * x;
      Eigen::VectorXf d = c.array().abs();

      std::vector<bool>        I_used(N, false); // Set of used indices
      std::vector<std::size_t> I(K); // Indices of the selected atoms (ordered sequence)

      for (std::size_t k=0; k<K; ++k) {
      	// Search for best matching atom
	float best_d       = -1.0f;
	std::size_t best_i = 0;
	for (std::size_t i=0; i<N; ++i) { 
	  if (!I_used[i]) {
	    float di = d[i];
	    if (di>best_d) {
	      best_d = di; best_i = i;
	    }
	  }
	}	  
	I[k] = best_i;
	I_used[best_i] = true;
	
	R(k,best_i) = u[best_i];

	// Project unused atoms onto subspace orthogonal to selected atom
	// and keep atoms normalized
	if (k+1!=K) {
	  for (std::size_t i=0; i<N; ++i) { 
	    if (!I_used[i]) {
	      // Update dactorization
	      R(k,i) = D.col(best_i).dot(D.col(i));
	      for (std::size_t n=0; n<k; ++n) {
		R(k,i) -= R(n,best_i) * R(n,i);
	      }
	      if (u[best_i] != 0.0f) R(k,i) /= u[best_i];
	      
	      c[i] = c[i]*u[i] - c[best_i] * R(k,i);
	      e[i] -= R(k,i)*R(k,i);
	      u[i] = std::sqrt(sl::abs(e[i]));
	      if (u[i] != 0.0f) c[i]/=u[i];
	      d[i] = sl::abs(c[i]);
	    }
	  }
	  // Ensure we select another atom
	  d[best_i] = 0.0f;
	}
      }


      // FIXME OPTIMIZE USING TRIANGULAR SOLVE FROM R
      // Assemble resulting system
      Eigen::MatrixXf Di(D.rows(),K);
      for (std::size_t k=0; k<K; ++k) {
	Di.col(k) = D.col(I[k]);
      }
      // Eigen::VectorXf gamma_I = Di.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(x-r);
      Eigen::VectorXf gamma_I = Di.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(x);
      for (std::size_t k=0; k<K; ++k) {
	gamma_idx[k] = I[k];
	gamma_val[k] = gamma_I[k];
      }
    }
    
    void ormp_in(std::vector<float>&        gamma_val,
		 std::vector<std::size_t>&  gamma_idx,
		 const Eigen::MatrixXf& D,
		 const Eigen::VectorXf& x,
		 std::size_t K) {
      const std::size_t N = std::size_t(D.cols());
      SL_USEVAR(N);
      assert(K<=N);
      assert(K>0);
      gamma_val.resize(K);
      gamma_idx.resize(K);
      ormp_in(&(gamma_val[0]), &(gamma_idx[0]), D, x, K);
    }
    
    void ormp_in(Eigen::VectorXf& gamma,
		 const Eigen::MatrixXf& D,
		 const Eigen::VectorXf& x,
		 std::size_t K) {
      std::vector<float>       gamma_val;
      std::vector<std::size_t> gamma_idx;
      ormp_in(gamma_val,
	      gamma_idx,
	      D, x, K);
      
      gamma.setZero(D.cols());
      for (std::size_t k=0; k<gamma_idx.size(); ++k) {
	gamma(gamma_idx[k]) = gamma_val[k];
      }
    } // ormp_in
      
  } // namespace tdm_sparse_coding

  
  // ----------------------------------------------------------------------
  // OMP with replacement
  // ----------------------------------------------------------------------

  namespace tdm_sparse_coding {

    // 
    void ompr_refine_in(float *gamma_val,
			std::size_t *gamma_idx,
			std::size_t K,
			const Eigen::MatrixXf& D,
			const Eigen::VectorXf& x,
			const Eigen::VectorXf& Dtx, // = trans(D)*x
			const Eigen::MatrixXf& DtD) {
      // If the solution is for K=0 or 1 no replacemente is possible
      // assuming that we start from a decent solution
      if (K<2) return;

      std::size_t Kmax = std::size_t(Dtx.rows());
      assert(K<=Kmax);

      const std::size_t T=2*K; // FIXME 
      const float tau = 1.0f; // FIXME

      std::vector<bool> is_support(Kmax, false);
      for (std::size_t k1=0; k1<K; ++k1) {
	is_support[gamma_idx[k1]] = true;
      }
      
      for (std::size_t t=0; t<T; ++t) {
	// r= x-Dg
	// similarity = max trans(d_k) * r
	// trans(D)r = trans(D)*x - trans(D)*D.g = Dtx - DtD * g  	
	Eigen::VectorXf alpha = Dtx;
	for (std::size_t k1=0; k1<K; ++k1) {
	  alpha -= DtD.col(gamma_idx[k1]) * gamma_val[k1];
	}
	std::size_t k_star = 0;
	float       a_star = 0.0f;
	// Find next most aligned vector
	for (std::size_t k1=0; k1<Kmax; ++k1) {
	  if (!is_support[k1]) {
	    const float a = sl::abs(alpha[k1]);
	    if (a>a_star) {
	      a_star = a; k_star = k1;
	    }
	  }
	}

	// Choose index in current support with smallest weight
	std::size_t ik_min = 0;
	float       zk_min = sl::abs(gamma_val[0] + tau * alpha[gamma_idx[0]]);
	for (std::size_t k1=1; k1<K; ++k1) {
	  const float z = sl::abs(gamma_val[k1] + tau * alpha[gamma_idx[k1]]);
	  if (z<zk_min) {
	    ik_min = k1; zk_min = z;
	  }
	}

	if (a_star<zk_min) {
	  // No improvement!
	  // std::cerr << "Keep " << ik_min << std::endl;
	  break;
	} else {
	  // std::cerr << "Replace " << ik_min << std::endl;
	  // Replace support and compute new solution
	  is_support[gamma_idx[ik_min]] = false; gamma_idx[ik_min] = k_star; is_support[k_star] = true;
	  least_squares_in(gamma_val,
			   gamma_idx,
			   D, x, K);
	}
      }
    }

    void ompr_refine_batch_in(std::vector<std::size_t>&  gamma_offset,
			      std::vector<std::size_t>&  gamma_sz,
			      std::vector<float>&        gamma_val,
			      std::vector<std::size_t>&  gamma_idx,
			      const Eigen::MatrixXf& D,
			      const Eigen::MatrixXf& DtD,
			      const std::vector<Eigen::VectorXf>& X) {
      const std::size_t X_count = X.size();
      //const std::size_t X_dim = D.rows();
      
      const Eigen::MatrixXf Dt = D.transpose();
      
#pragma omp parallel for schedule(static)
      for (std::size_t i=0; i<X_count; ++i) {
	if (gamma_sz[i] > 2) {
	  const Eigen::VectorXf& x_i = X[i];
	  
	  const Eigen::VectorXf Dtx_i = Dt*x_i;
	  
	  ompr_refine_in(&(gamma_val[gamma_offset[i]]),
			 &(gamma_idx[gamma_offset[i]]),
			 gamma_sz[gamma_offset[i]],
			 D, x_i, Dtx_i, DtD);
	}
      }			  
    }
    
  } // namespace  tdm_sparse_coding

  // ----------------------------------------------------------------------
  // OMP with replacement
  // ----------------------------------------------------------------------

  namespace  tdm_sparse_coding {
 
    void imp_refine_in(float *gamma_val,
		       std::size_t *gamma_idx,
		       std::size_t K,
		       const Eigen::MatrixXf& D,
		       const Eigen::VectorXf& x,
		       const Eigen::VectorXf& Dtx, // = trans(D)*x
		       const Eigen::MatrixXf& /*DtD*/) {
      // If the solution is for K=0 or 1 no replacemente is possible
      // assuming that we start from a decent solution
      if (K<2) return;

      std::size_t Kmax = std::size_t(Dtx.rows());
      assert(K<=Kmax);

      // Hack -- we implement full imp instead of just refining

      // Forward pass
      Eigen::VectorXf r = x;
	
      for (std::size_t k=0; k<K; ++k) {
	
	std::size_t k_star = 0;
	float       a_star = D.col(0).dot(r);
	// Find next most aligned vector
	for (std::size_t k1=1; k1<Kmax; ++k1) {
	  const float a = D.col(k1).dot(r);
	  if (sl::abs(a)>sl::abs(a_star)) {
	    a_star = a; k_star = k1;
	  }
	}
	gamma_idx[k] = k_star;
	gamma_val[k] = a_star;
	r -= gamma_val[k] * D.col(gamma_idx[k]);
      }
      
    
      // Backward pass
      for (std::size_t k=0; k<K; ++k) {
	// Remove one element from support
	r += gamma_val[k] * D.col(gamma_idx[k]);

	// Find most aligned with current residual
	std::size_t k_star = 0;
	float       a_star = D.col(0).dot(r);
	// Find next most aligned vector
	for (std::size_t k1=1; k1<Kmax; ++k1) {
	  const float a = D.col(k1).dot(r);
	  if (sl::abs(a)>sl::abs(a_star)) {
	    a_star = a; k_star = k1;
	  }
	}

	// Replace element
	gamma_idx[k] = k_star;
	gamma_val[k] = a_star;
	r -= gamma_val[k] * D.col(gamma_idx[k]);
      }

      // Final least squares
      least_squares_in(gamma_val,
		       gamma_idx,
		       D, x, K);
    }

    void imp_refine_batch_in(std::vector<std::size_t>&  gamma_offset,
			      std::vector<std::size_t>&  gamma_sz,
			      std::vector<float>&        gamma_val,
			      std::vector<std::size_t>&  gamma_idx,
			      const Eigen::MatrixXf& D,
			      const Eigen::MatrixXf& DtD,
			      const std::vector<Eigen::VectorXf>& X) {
      const std::size_t X_count = X.size();
      //const std::size_t X_dim = D.rows();
      
      const Eigen::MatrixXf Dt = D.transpose();
      
#pragma omp parallel for schedule(static)
      for (std::size_t i=0; i<X_count; ++i) {
	if (gamma_sz[i] > 2) {
	  const Eigen::VectorXf& x_i = X[i];
	  
	  const Eigen::VectorXf Dtx_i = Dt*x_i;
	  
	  imp_refine_in(&(gamma_val[gamma_offset[i]]),
			&(gamma_idx[gamma_offset[i]]),
			gamma_sz[gamma_offset[i]],
			D, x_i, Dtx_i, DtD);
	}
      }			  
    }
    
  } // namespace tdm_sparse_coding


  
} // namespace vic

