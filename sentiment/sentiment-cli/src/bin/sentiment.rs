use clap::{crate_version, App, Arg, ArgMatches};
use onnxruntime::{
    environment::Environment,
    ndarray::{Array, Axis},
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel, LoggingLevel,
};
use ordered_float::OrderedFloat;
use tokenizers::{
    models::wordpiece::WordPiece,
    normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer,
    processors::bert::BertProcessing,
    tokenizer::{EncodeInput, Tokenizer},
    utils::{
        padding::{PaddingDirection, PaddingParams, PaddingStrategy},
        truncation::{TruncationParams, TruncationStrategy},
    },
};

fn main() -> Result<(), onnxruntime::error::OrtError> {
    let args = parse_args();

    let vocab_path = args
        .value_of("vocab-path")
        .expect("tokenizer vocab.txt file is required");

    let model = WordPiece::from_files(vocab_path)
        .build()
        .expect("Failed to build wordpiece from file");

    let text = args.value_of("text").expect("text is required");

    let mut tok = Tokenizer::new(Box::new(model));

    tok.with_truncation(Some(TruncationParams {
        max_length: 128,
        strategy: TruncationStrategy::OnlyFirst,
        stride: 1,
    }));

    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(128),
        direction: PaddingDirection::Right,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".into(),
    }));

    tok.with_pre_tokenizer(Box::new(BertPreTokenizer));

    tok.with_normalizer(Box::new(BertNormalizer::new(true, true, true, false)));

    tok.with_post_processor(Box::new(BertProcessing::new(
        ("[SEP]".into(), 102),
        ("[CLS]".into(), 101),
    )));

    let encoded = tok.encode(EncodeInput::Single(text.into()), true).unwrap();

    let input_ids = encoded.get_ids().to_vec();
    let attn_mask = encoded.get_attention_mask().to_vec();

    let path = args.value_of("model-path").expect("model-path is required");
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file(path)?;

    let arr = Array::from_iter(input_ids.clone().into_iter().map(|i| i as i64))
        .into_shape((1, input_ids.len()))
        .expect("failed to reshape");

    let attention_mask = Array::from_iter(attn_mask.into_iter().map(|i| i as i64))
        .into_shape((1, input_ids.len()))
        .unwrap();

    let type_ids = arr.map(|_| 0);

    let output: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![arr, attention_mask, type_ids])?;
    for out in output {
        let predictions = out.softmax(Axis(1));
        let (idx, prob) = predictions
            .iter()
            .enumerate()
            .max_by_key(|(_, prob)| OrderedFloat(**prob))
            .unwrap();
        match idx {
            0 => println!("{0} : Negative {1:.5}", text, prob),
            1 => println!("{0} : Positive {1:.5}", text, prob),
            2 => println!("{0} : Neutral {1:.5}", text, prob),
            3 => println!("{0} : Mixed {1:.5}", text, prob),
            a => panic!("Received unexpected index: {}", a),
        }
    }

    Ok(())
}

fn parse_args<'a>() -> ArgMatches<'a> {
    App::new("sentiment-cli")
        .version(crate_version!())
        .arg(Arg::with_name("model-path").required(true))
        .arg(
            Arg::with_name("vocab-path")
                .long("tokenizer-vocab")
                .short("t")
                .takes_value(true)
                .number_of_values(1)
                .required(true),
        )
        .arg(Arg::with_name("text").required(true))
        .get_matches()
}
