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
#ifndef VIC_TDM_SPARSE_CODING_STREAMING_CORESET_BUILDER_UNIFORM_HPP
#define VIC_TDM_SPARSE_CODING_STREAMING_CORESET_BUILDER_UNIFORM_HPP

#include <vic/tdm_sparse_coding/streaming_coreset_builder.hpp>
#include <sl/random.hpp>

namespace vic {
  
  namespace tdm_sparse_coding {
    
    /*
    ** Streaming coreset builder based on a simple uniform
    ** sampling strategy. 
    */
    class streaming_coreset_builder_uniform: public streaming_coreset_builder {
    protected:
      typedef streaming_coreset_builder super_t;
    protected:
      sl::random::uniform<float> rng_;
      sl::random::std_irng_t    irng_;

      sl::uint64_t    item_count_;
      
    public: // Creation & destruction
      
      streaming_coreset_builder_uniform();

      virtual ~streaming_coreset_builder_uniform();

      virtual void clear();

    public: // Params
     
    public: // Output

    public: // Input: Streaming interface
      
      virtual void begin();
      
      virtual void end();

    protected:

      virtual void inner_put(const Eigen::VectorXf& y);
      
    }; // class streaming_coreset_builder_uniform
  } // namespace tdm_sparse_coding
} // namespace vic

#endif
