/*!
 *  Copyright (c) 2023 by Contributors
 * \file sentencepiece_tokenizer_disabled.cc
 * \brief Disabled SentencePiece tokenizer factory
 */
#include <tokenizers_cpp.h>

namespace tokenizers {

std::unique_ptr<Tokenizer> Tokenizer::FromBlobSentencePiece(const std::string& model_blob) {
  (void)model_blob;
  return nullptr;
}

}  // namespace tokenizers
