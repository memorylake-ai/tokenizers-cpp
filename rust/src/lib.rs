// A C wrapper around the Hugging Face tokenizers library.
use ahash::AHashMap;
use serde_json::Value;
use std::any::Any;
use std::convert::TryFrom;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::str::FromStr;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{AddedVocabulary, Tokenizer};

const ENCODE_FLAG_IGNORE_SPECIAL_TOKENS: u32 = 1 << 0;
const ENCODE_FLAG_IGNORE_ADDED_TOKENS: u32 = 1 << 1;
const ENCODE_FLAG_MASK: u32 = ENCODE_FLAG_IGNORE_SPECIAL_TOKENS | ENCODE_FLAG_IGNORE_ADDED_TOKENS;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizerStatus {
    Ok = 0,
    InvalidArgument = 1,
    Unsupported = 2,
    Internal = 3,
}

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut u32,
    len: usize,
}

#[repr(C)]
pub struct TokenizerErrorMessage {
    data: *mut u8,
    len: usize,
}

#[derive(Debug)]
enum FfiError {
    InvalidArgument(String),
    Unsupported(String),
    Internal(String),
}

impl FfiError {
    fn status(&self) -> TokenizerStatus {
        match self {
            Self::InvalidArgument(_) => TokenizerStatus::InvalidArgument,
            Self::Unsupported(_) => TokenizerStatus::Unsupported,
            Self::Internal(_) => TokenizerStatus::Internal,
        }
    }

    fn into_message(self) -> String {
        match self {
            Self::InvalidArgument(message)
            | Self::Unsupported(message)
            | Self::Internal(message) => message,
        }
    }
}

// JSON tokenizers get immutable policy-specific views at load time. Encoding can then select a
// view per call without mutating shared state or cloning the model on the request path.
struct EncodeVariants {
    ignore_special_tokens: Tokenizer,
    ignore_added_tokens: Tokenizer,
}

struct TokenizerWrapper {
    tokenizer: Tokenizer,
    encode_variants: Option<EncodeVariants>,
    decode_str: String,
    id_to_token_result: String,
}

pub type Vocab = AHashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

impl TokenizerWrapper {
    fn from_str(json: &str) -> Result<TokenizerWrapper, FfiError> {
        let tokenizer = Tokenizer::from_str(json).map_err(|error| {
            FfiError::InvalidArgument(format!("Failed to parse tokenizer JSON: {error}"))
        })?;

        let mut ignore_special_tokens = tokenizer.clone();
        ignore_special_tokens.set_encode_special_tokens(true);

        let mut ignore_added_tokens = tokenizer.clone();
        ignore_added_tokens.with_added_vocabulary(AddedVocabulary::new());

        Ok(TokenizerWrapper {
            tokenizer,
            encode_variants: Some(EncodeVariants {
                ignore_special_tokens,
                ignore_added_tokens,
            }),
            decode_str: String::new(),
            id_to_token_result: String::new(),
        })
    }

    fn byte_level_bpe_from_str(
        vocab: &str,
        merges: &str,
        added_tokens: &str,
    ) -> Result<TokenizerWrapper, FfiError> {
        let vocab_json: Value = serde_json::from_str(vocab).map_err(|error| {
            FfiError::InvalidArgument(format!("Failed to parse BPE vocabulary JSON: {error}"))
        })?;
        let mut vocab = ahash::AHashMap::new();
        match vocab_json {
            Value::Object(entries) => {
                for (token, id) in entries {
                    let id = id.as_u64().ok_or_else(|| {
                        FfiError::InvalidArgument(format!(
                            "Vocabulary id for token {token:?} is not an unsigned integer"
                        ))
                    })?;
                    let id = u32::try_from(id).map_err(|_| {
                        FfiError::InvalidArgument(format!(
                            "Vocabulary id for token {token:?} exceeds u32"
                        ))
                    })?;
                    vocab.insert(token, id);
                }
            }
            _ => {
                return Err(FfiError::InvalidArgument(
                    "BPE vocabulary JSON must be an object".to_string(),
                ))
            }
        }

        if !added_tokens.is_empty() {
            let added_tokens_json: Value = serde_json::from_str(added_tokens).map_err(|error| {
                FfiError::InvalidArgument(format!("Failed to parse added-token JSON: {error}"))
            })?;
            match added_tokens_json {
                Value::Object(entries) => {
                    for (token, id) in entries {
                        let id = id.as_u64().ok_or_else(|| {
                            FfiError::InvalidArgument(format!(
                                "Added-token id for token {token:?} is not an unsigned integer"
                            ))
                        })?;
                        let id = u32::try_from(id).map_err(|_| {
                            FfiError::InvalidArgument(format!(
                                "Added-token id for token {token:?} exceeds u32"
                            ))
                        })?;
                        vocab.insert(token, id);
                    }
                }
                _ => {
                    return Err(FfiError::InvalidArgument(
                        "Added-token JSON must be an object".to_string(),
                    ))
                }
            }
        }

        let merges = merges
            .lines()
            .filter(|line| !line.starts_with("#version"))
            .map(|line| {
                let parts = line.split(' ').collect::<Vec<_>>();
                if parts.len() != 2 {
                    return Err(FfiError::InvalidArgument(format!(
                        "Invalid BPE merge line: {line:?}"
                    )));
                }
                Ok((parts[0].to_string(), parts[1].to_string()))
            })
            .collect::<Result<Vec<(String, String)>, FfiError>>()?;
        let byte_level = ByteLevel::new(
            /* add_prefix_space = */ false, /* trim_offsets = */ false,
            /* use_regex = */ false,
        );
        let mut tokenizer = Tokenizer::new(BPE::new(vocab, merges));
        tokenizer
            .with_pre_tokenizer(Some(byte_level))
            .with_decoder(Some(byte_level));

        // This legacy factory inserts its added-token map directly into the BPE model vocabulary,
        // so it cannot distinguish ordinary model tokens from added tokens at encode time.
        Ok(TokenizerWrapper {
            tokenizer,
            encode_variants: None,
            decode_str: String::new(),
            id_to_token_result: String::new(),
        })
    }

    fn tokenizer_for_encode(&self, flags: u32) -> Result<&Tokenizer, FfiError> {
        if flags & !ENCODE_FLAG_MASK != 0 {
            return Err(FfiError::InvalidArgument(format!(
                "Unknown encode flag bits: 0x{:x}",
                flags & !ENCODE_FLAG_MASK
            )));
        }

        if flags == 0 {
            return Ok(&self.tokenizer);
        }

        let variants = self.encode_variants.as_ref().ok_or_else(|| {
            FfiError::Unsupported(
                "Added-token ignore flags require a Hugging Face JSON tokenizer".to_string(),
            )
        })?;
        if flags & ENCODE_FLAG_IGNORE_ADDED_TOKENS != 0 {
            Ok(&variants.ignore_added_tokens)
        } else {
            Ok(&variants.ignore_special_tokens)
        }
    }

    fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
        flags: u32,
    ) -> Result<Vec<u32>, FfiError> {
        let tokenizer = self.tokenizer_for_encode(flags)?;
        let encoded = tokenizer
            .encode(text, add_special_tokens)
            .map_err(|error| {
                FfiError::Internal(format!("Hugging Face tokenizer encode failed: {error}"))
            })?;
        Ok(encoded.get_ids().to_vec())
    }

    fn encode_batch(
        &self,
        texts: Vec<&str>,
        add_special_tokens: bool,
        flags: u32,
    ) -> Result<Vec<Vec<u32>>, FfiError> {
        let tokenizer = self.tokenizer_for_encode(flags)?;
        let results = tokenizer
            .encode_batch(texts, add_special_tokens)
            .map_err(|error| {
                FfiError::Internal(format!(
                    "Hugging Face tokenizer batch encode failed: {error}"
                ))
            })?
            .into_iter()
            .map(|encoded| encoded.get_ids().to_vec())
            .collect();
        Ok(results)
    }

    fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) {
        self.decode_str = self
            .tokenizer
            .decode(ids, skip_special_tokens)
            .unwrap_or_default();
    }
}

unsafe fn input_str<'a>(data: *const u8, len: usize, name: &str) -> Result<&'a str, FfiError> {
    if len == 0 {
        return Ok("");
    }
    if data.is_null() {
        return Err(FfiError::InvalidArgument(format!(
            "{name} is null with non-zero length"
        )));
    }
    std::str::from_utf8(std::slice::from_raw_parts(data, len))
        .map_err(|error| FfiError::InvalidArgument(format!("{name} is not valid UTF-8: {error}")))
}

unsafe fn wrapper_from_handle<'a>(handle: *mut c_void) -> Result<&'a TokenizerWrapper, FfiError> {
    if handle.is_null() {
        return Err(FfiError::InvalidArgument(
            "Tokenizer handle is null".to_string(),
        ));
    }
    Ok(&*(handle as *const TokenizerWrapper))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("Rust tokenizer panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("Rust tokenizer panicked: {message}")
    } else {
        "Rust tokenizer panicked with a non-string payload".to_string()
    }
}

unsafe fn write_error_message(out_error: *mut TokenizerErrorMessage, message: String) {
    if out_error.is_null() {
        return;
    }
    let bytes = message.into_bytes().into_boxed_slice();
    let len = bytes.len();
    let data = Box::into_raw(bytes) as *mut u8;
    *out_error = TokenizerErrorMessage { data, len };
}

fn ffi_call<F>(out_error: *mut TokenizerErrorMessage, action: F) -> TokenizerStatus
where
    F: FnOnce() -> Result<(), FfiError>,
{
    unsafe {
        if !out_error.is_null() {
            *out_error = TokenizerErrorMessage {
                data: ptr::null_mut(),
                len: 0,
            };
        }
    }

    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => TokenizerStatus::Ok,
        Ok(Err(error)) => {
            let status = error.status();
            unsafe {
                write_error_message(out_error, error.into_message());
            }
            status
        }
        Err(payload) => {
            unsafe {
                write_error_message(out_error, panic_message(payload));
            }
            TokenizerStatus::Internal
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(
    input_json: *const u8,
    len: usize,
    out_handle: *mut *mut c_void,
    out_error: *mut TokenizerErrorMessage,
) -> TokenizerStatus {
    ffi_call(out_error, || unsafe {
        if out_handle.is_null() {
            return Err(FfiError::InvalidArgument(
                "Output tokenizer handle is null".to_string(),
            ));
        }
        *out_handle = ptr::null_mut();
        let json = input_str(input_json, len, "Tokenizer JSON")?;
        let wrapper = TokenizerWrapper::from_str(json)?;
        *out_handle = Box::into_raw(Box::new(wrapper)) as *mut c_void;
        Ok(())
    })
}

#[no_mangle]
extern "C" fn byte_level_bpe_tokenizers_new_from_str(
    input_vocab: *const u8,
    vocab_len: usize,
    input_merges: *const u8,
    merges_len: usize,
    input_added_tokens: *const u8,
    added_tokens_len: usize,
    out_handle: *mut *mut c_void,
    out_error: *mut TokenizerErrorMessage,
) -> TokenizerStatus {
    ffi_call(out_error, || unsafe {
        if out_handle.is_null() {
            return Err(FfiError::InvalidArgument(
                "Output tokenizer handle is null".to_string(),
            ));
        }
        *out_handle = ptr::null_mut();
        let vocab = input_str(input_vocab, vocab_len, "BPE vocabulary")?;
        let merges = input_str(input_merges, merges_len, "BPE merges")?;
        let added_tokens = input_str(input_added_tokens, added_tokens_len, "BPE added-token map")?;
        let wrapper = TokenizerWrapper::byte_level_bpe_from_str(vocab, merges, added_tokens)?;
        *out_handle = Box::into_raw(Box::new(wrapper)) as *mut c_void;
        Ok(())
    })
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut c_void,
    input_text: *const u8,
    len: usize,
    add_special_tokens: i32,
    flags: u32,
    out_result: *mut TokenizerEncodeResult,
    out_error: *mut TokenizerErrorMessage,
) -> TokenizerStatus {
    ffi_call(out_error, || unsafe {
        if out_result.is_null() {
            return Err(FfiError::InvalidArgument(
                "Output encode result is null".to_string(),
            ));
        }
        *out_result = TokenizerEncodeResult {
            token_ids: ptr::null_mut(),
            len: 0,
        };
        let wrapper = wrapper_from_handle(handle)?;
        let text = input_str(input_text, len, "Encode input")?;
        let encoded = wrapper.encode(text, add_special_tokens != 0, flags)?;
        let encoded = encoded.into_boxed_slice();
        let len = encoded.len();
        let token_ids = Box::into_raw(encoded) as *mut u32;
        *out_result = TokenizerEncodeResult { token_ids, len };
        Ok(())
    })
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch(
    handle: *mut c_void,
    input_texts: *const *const u8,
    input_lens: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    flags: u32,
    out_results: *mut TokenizerEncodeResult,
    out_error: *mut TokenizerErrorMessage,
) -> TokenizerStatus {
    ffi_call(out_error, || unsafe {
        if num_seqs == 0 {
            let wrapper = wrapper_from_handle(handle)?;
            wrapper.encode_batch(Vec::new(), add_special_tokens != 0, flags)?;
            return Ok(());
        }
        if input_texts.is_null() || input_lens.is_null() || out_results.is_null() {
            return Err(FfiError::InvalidArgument(
                "Batch input or output array is null".to_string(),
            ));
        }

        for index in 0..num_seqs {
            *out_results.add(index) = TokenizerEncodeResult {
                token_ids: ptr::null_mut(),
                len: 0,
            };
        }

        let wrapper = wrapper_from_handle(handle)?;
        let mut texts = Vec::with_capacity(num_seqs);
        for index in 0..num_seqs {
            texts.push(input_str(
                *input_texts.add(index),
                *input_lens.add(index),
                "Batch encode input",
            )?);
        }
        let encoded_batch = wrapper.encode_batch(texts, add_special_tokens != 0, flags)?;
        let boxed_results = encoded_batch
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect::<Vec<_>>();

        // All allocations are complete before ownership is transferred to C, so an allocation
        // failure cannot leak a partially written result array.
        for (index, encoded) in boxed_results.into_iter().enumerate() {
            let len = encoded.len();
            let token_ids = Box::into_raw(encoded) as *mut u32;
            *out_results.add(index) = TokenizerEncodeResult { token_ids, len };
        }
        Ok(())
    })
}

#[no_mangle]
extern "C" fn tokenizers_free_encode_results(results: *mut TokenizerEncodeResult, num_seqs: usize) {
    if num_seqs == 0 || results.is_null() {
        return;
    }
    unsafe {
        for result in std::slice::from_raw_parts_mut(results, num_seqs) {
            if !result.token_ids.is_null() {
                drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
                    result.token_ids,
                    result.len,
                )));
                result.token_ids = ptr::null_mut();
                result.len = 0;
            }
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_free_error_message(error: *mut TokenizerErrorMessage) {
    if error.is_null() {
        return;
    }
    unsafe {
        if !(*error).data.is_null() {
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
                (*error).data,
                (*error).len,
            )));
            (*error).data = ptr::null_mut();
            (*error).len = 0;
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32,
) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let ids = if len == 0 {
            &[]
        } else if input_ids.is_null() {
            return;
        } else {
            std::slice::from_raw_parts(input_ids, len)
        };
        (*handle).decode(ids, skip_special_tokens != 0);
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    if handle.is_null() || out_cstr.is_null() || out_len.is_null() {
        return;
    }
    unsafe {
        let decode_str = &mut (*handle).decode_str;
        *out_cstr = decode_str.as_mut_ptr();
        *out_len = decode_str.len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_vocab_size(handle: *mut TokenizerWrapper, size: *mut usize) {
    if handle.is_null() || size.is_null() {
        return;
    }
    unsafe {
        *size = (*handle).tokenizer.get_vocab_size(true);
    }
}

#[no_mangle]
extern "C" fn tokenizers_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    if handle.is_null() || out_cstr.is_null() || out_len.is_null() {
        return;
    }
    unsafe {
        let token = (*handle).tokenizer.id_to_token(id);
        (*handle).id_to_token_result = token.unwrap_or_default();

        let id_to_token_result = &mut (*handle).id_to_token_result;
        *out_cstr = id_to_token_result.as_mut_ptr();
        *out_len = id_to_token_result.len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize,
    out_id: *mut i32,
) {
    if handle.is_null() || out_id.is_null() {
        return;
    }
    unsafe {
        let token = match input_str(token, len, "Token") {
            Ok(token) => token,
            Err(_) => {
                *out_id = -1;
                return;
            }
        };
        let id = (*handle).tokenizer.token_to_id(token);
        *out_id = id.and_then(|value| i32::try_from(value).ok()).unwrap_or(-1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    const TEST_TOKENIZER_JSON: &str = r#"
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 11,
      "content": "<special>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 12,
      "content": "<added>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<special>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [
      {"Sequence": {"id": "A", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}}
    ],
    "special_tokens": {
      "<special>": {
        "id": "<special>",
        "ids": [11],
        "tokens": ["<special>"]
      }
    }
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "[UNK]",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {
      "[UNK]": 0,
      "<": 1,
      ">": 2,
      "s": 3,
      "p": 4,
      "e": 5,
      "c": 6,
      "i": 7,
      "a": 8,
      "l": 9,
      "d": 10
    },
    "merges": []
  }
}
"#;

    const ORDINARY_SPECIAL: &[u32] = &[1, 3, 4, 5, 6, 7, 8, 9, 2];
    const ORDINARY_ADDED: &[u32] = &[1, 8, 10, 10, 5, 10, 2];

    #[test]
    fn encode_flags_select_added_token_policy() {
        let mut wrapper = TokenizerWrapper::from_str(TEST_TOKENIZER_JSON).unwrap();

        assert_eq!(wrapper.encode("<special>", false, 0).unwrap(), vec![11]);
        assert_eq!(wrapper.encode("<added>", false, 0).unwrap(), vec![12]);

        assert_eq!(
            wrapper
                .encode("<special>", false, ENCODE_FLAG_IGNORE_SPECIAL_TOKENS)
                .unwrap(),
            ORDINARY_SPECIAL
        );
        assert_eq!(
            wrapper
                .encode("<added>", false, ENCODE_FLAG_IGNORE_SPECIAL_TOKENS)
                .unwrap(),
            vec![12]
        );

        let all_added_flags = [
            ENCODE_FLAG_IGNORE_ADDED_TOKENS,
            ENCODE_FLAG_IGNORE_SPECIAL_TOKENS | ENCODE_FLAG_IGNORE_ADDED_TOKENS,
        ];
        for flags in all_added_flags {
            assert_eq!(
                wrapper.encode("<special>", false, flags).unwrap(),
                ORDINARY_SPECIAL
            );
            let ordinary_added = wrapper.encode("<added>", false, flags).unwrap();
            assert_eq!(ordinary_added, ORDINARY_ADDED);
            wrapper.decode(&ordinary_added, false);
            assert_eq!(wrapper.decode_str, "<added>");
        }
    }

    #[test]
    fn add_special_tokens_remains_orthogonal() {
        let wrapper = TokenizerWrapper::from_str(TEST_TOKENIZER_JSON).unwrap();
        let encoded = wrapper
            .encode("<added>", true, ENCODE_FLAG_IGNORE_ADDED_TOKENS)
            .unwrap();
        assert_eq!(encoded[0], 11);
        assert_eq!(&encoded[1..], ORDINARY_ADDED);
    }

    #[test]
    fn batch_matches_single_and_handles_empty_inputs() {
        let wrapper = TokenizerWrapper::from_str(TEST_TOKENIZER_JSON).unwrap();
        let texts = vec!["<special><added>", "", "<added><added>"];
        for flags in [
            0,
            ENCODE_FLAG_IGNORE_SPECIAL_TOKENS,
            ENCODE_FLAG_IGNORE_ADDED_TOKENS,
        ] {
            let batch = wrapper.encode_batch(texts.clone(), false, flags).unwrap();
            let singles = texts
                .iter()
                .map(|text| wrapper.encode(text, false, flags).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(batch, singles);
        }
        assert!(wrapper
            .encode_batch(Vec::new(), false, 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn invalid_flags_fail_without_changing_default_behavior() {
        let wrapper = TokenizerWrapper::from_str(TEST_TOKENIZER_JSON).unwrap();
        assert!(matches!(
            wrapper.encode("<special>", false, 1 << 31),
            Err(FfiError::InvalidArgument(_))
        ));
        assert!(matches!(
            wrapper.encode_batch(Vec::new(), false, 1 << 31),
            Err(FfiError::InvalidArgument(_))
        ));
        assert_eq!(wrapper.encode("<special>", false, 0).unwrap(), vec![11]);
    }

    #[test]
    fn concurrent_policies_are_isolated() {
        let wrapper = Arc::new(TokenizerWrapper::from_str(TEST_TOKENIZER_JSON).unwrap());
        let barrier = Arc::new(Barrier::new(4));
        let policies = vec![
            (0, vec![11, 12]),
            (
                ENCODE_FLAG_IGNORE_SPECIAL_TOKENS,
                [ORDINARY_SPECIAL, &[12]].concat(),
            ),
            (
                ENCODE_FLAG_IGNORE_ADDED_TOKENS,
                [ORDINARY_SPECIAL, ORDINARY_ADDED].concat(),
            ),
        ];

        let workers = policies
            .into_iter()
            .map(|(flags, expected)| {
                let wrapper = Arc::clone(&wrapper);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..100 {
                        assert_eq!(
                            wrapper.encode("<special><added>", false, flags).unwrap(),
                            expected
                        );
                        assert_eq!(
                            wrapper
                                .encode_batch(
                                    vec!["<special><added>", "<special><added>"],
                                    false,
                                    flags,
                                )
                                .unwrap(),
                            vec![expected.clone(), expected.clone()]
                        );
                    }
                })
            })
            .collect::<Vec<_>>();

        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            wrapper.encode("<special><added>", false, 0).unwrap(),
            vec![11, 12]
        );
    }
}
