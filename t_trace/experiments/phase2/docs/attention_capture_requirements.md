Here’s a concise, ready-to-record note for your project documentation:
⚠️ Critical Pre-Condition: Hugging Face Attention Capture

Requirement: Must enable output_attentions=True before first forward pass for attention weights to be computable.

# METHOD 1: During model loading (RECOMMENDED)
from transformers import BertConfig, BertModel

config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True)
model = BertModel.from_pretrained('bert-base-uncased', config=config)

# METHOD 2: Post-load (MUST be BEFORE first inference)
model = BertModel.from_pretrained('bert-base-uncased')
model.config.output_attentions = True  # ⚠️ Setting AFTER forward pass = NO ATTENTION WEIGHTS

Why this is non-negotiable:

    Hugging Face transformers skip attention probability computation entirely when output_attentions=False (default)
    Your hooks correctly extract attention_probs from submodules – but nothing exists to extract if computation was skipped
    This is a framework limitation, not a hook implementation defect
    Without this config: attention_weights: [] persists regardless of Gap 1 fixes

Validation checklist before Winograd experiments:

assert model.config.output_attentions == True, "ATTENTION CAPTURE WILL FAIL"
outputs = model(**inputs)
assert hasattr(outputs, 'attentions'), "Model didn't compute attention weights"
assert outputs.attentions is not None, "Attention tuple is empty"

    💡 Pro Tip: Add this assertion to your enable_logging() wrapper for transformer models to fail-fast during development:

    if "bert" in model.__class__.__name__.lower() and not getattr(model.config, "output_attentions", False):
        raise RuntimeError("Hugging Face model requires output_attentions=True for attention logging")

Record this in:
✅ docs/attention_capture_requirements.md
✅ Experiment 1 validation script header comments
✅ Phase 2 experimental protocol checklist (Section 4.2)
