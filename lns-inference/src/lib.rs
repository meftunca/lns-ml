pub mod compat;
pub mod engine;
pub mod model_config;
pub mod sampler;
pub mod tokenizer;
pub mod transformer;

pub use compat::{inspect_compatibility, CompatibilityReport, SupportTier, TemplateRecommendation};
pub use engine::{ForwardTrace, InferenceEngine, LayerTrace};
pub use model_config::{
    inspect_architecture, parse_model_config, ArchitectureSpec, ModelFamily, NormalizedModelConfig,
    TextLayerKind,
};
pub use tokenizer::Tokenizer;
pub use transformer::{
    HybridAttentionWeights, HybridLayerWeights, HybridTransformerWeights, MoeConfig,
    MoeExpertWeights, MoeMlpWeights, TransformerConfig, TransformerWeights,
};
