#!/usr/bin/env python3
r"""
Experiment 2 Ablation: Multi-Layer Patching Robustness Check
=====================================================================
Tests whether single-layer patching artifactually depresses CSVR.
Patches contiguous 3-layer windows [l-1, l, l+1] around each token's peak layer.
Outputs: exp2_multilayer_seed{N}_results.json + aggregated JSON + LaTeX table
"""
import os, sys, json, time, warnings, argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import ttest_ind
import torch
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import captum.attr as attr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from t_trace.logging_engine import enable_logging

warnings.filterwarnings("ignore")

CONFIG = {
    "models": [
        {"name": "textattack/bert-base-uncased-SST-2", "type": "bert", "tokenizer": BertTokenizer},
        {"name": "textattack/roberta-base-SST-2", "type": "roberta", "tokenizer": RobertaTokenizer}
    ],
    "dataset": "glue", "dataset_config": "sst2", "num_samples": 200,
    "batch_size": 8, "top_k_attributions": 5, "patching_threshold": 0.15,
    "window_size": 3, "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_dir": Path(__file__).parent / "results",
    "mtrace_mode": "development", "seeds": [42, 123, 456, 789, 1011]
}

# ================= STAT UTILS (Reused) =================
def welchs_t_test(v1, v2=None):
    if v2 is None: v2 = [0.0]*len(v1)
    t, p = ttest_ind(v1, v2, equal_var=False)
    n1, n2 = len(v1), len(v2)
    var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2)) if (n1+n2-2)>0 else 1.0
    d = (np.mean(v1)-np.mean(v2))/pooled if pooled>0 else 0.0
    return {"p_value": float(p), "effect_size": float(d)}

def format_p(p): return r"$p<0.001$" if p<0.001 else f"$p={p:.3f}$"
def fmt(mean, std, p, n): return f"${mean:.3f}\\pm{std:.3f}$({format_p(p)},$n={n}$)"

# ================= MULTI-LAYER PATCHER =================
class MultiLayerPatcher:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.baseline = {}
        
    def collect_baseline(self, neutral_texts):
        layer_acts = {i:[] for i in range(12)}
        def hook(idx):
            def fn(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                layer_acts[idx].append(h.mean(dim=1).detach().cpu().numpy())
            return fn
        layers = self.model.bert.encoder.layer if hasattr(self.model,'bert') else self.model.roberta.encoder.layer
        handles = [layers[i].register_forward_hook(hook(i)) for i in range(12)]
        for t in neutral_texts[:20]:
            inp = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad(): self.model(**inp, output_hidden_states=True)
        for h in handles: h.remove()
        for i in range(12):
            if layer_acts[i]:
                stacked = np.concatenate(layer_acts[i], axis=0)
                self.baseline[i] = np.mean(stacked, axis=0, keepdims=True)

    def patch_window(self, inputs, center_layer, window=3):
        start = max(0, center_layer - window//2)
        end = min(12, start + window)
        layers = self.model.bert.encoder.layer if hasattr(self.model,'bert') else self.model.roberta.encoder.layer
        
        # Baseline
        baselines = {}
        for i in range(start, end):
            if i in self.baseline: 
                baselines[i] = torch.from_numpy(self.baseline[i]).float().to(self.device)
            else: 
                baselines[i] = None
                
        if not baselines: return 0.0
        
        # Original output
        with torch.no_grad(): orig = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        
        # Patched output
        def patch_hook(idx, baseline):
            def fn(m, i, o):
                if baseline is None: return o
                
                # 1. Handle tuple vs single tensor outputs (transformers version variance)
                hs = o[0] if isinstance(o, tuple) else o
                
                # 2. Restore missing batch dimension if squeezed (2D -> 3D)
                if hs.dim() == 2:
                    hs = hs.unsqueeze(0)
                    
                bs, seq, hid = hs.shape
                patched_hs = baseline.expand(bs, seq, hid)
                
                # 3. Return matching structure
                return (patched_hs,) + o[1:] if isinstance(o, tuple) else patched_hs
            return fn
            
        handles = [layers[i].register_forward_hook(patch_hook(i, baselines.get(i))) 
                   for i in range(start, end)]
        with torch.no_grad(): patched = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        for h in handles: h.remove()
        
        return torch.abs(orig - patched).max().item()

    def verify_window(self, inputs, peak_layer):
        if peak_layer not in self.baseline: return False
        return self.patch_window(inputs, peak_layer, CONFIG["window_size"]) > CONFIG["patching_threshold"]

# ================= METRICS =================
def get_token_peak_layers(logs, tokens):
    peaks = {}
    intensity = np.zeros((len(tokens), 12))
    for log in logs:
        if log.get("event_type") != "forward": continue
        idx = log["internal_states"].get("layer_index", -1)
        if not (0 <= idx < 12): continue
        attn = log["internal_states"].get("attention_weights", [])
        if isinstance(attn, dict) and "sparse_values" in attn:
            vals, shape = attn["sparse_values"], attn.get("shape", [1,1,len(tokens),len(tokens)])
            mat = np.zeros(np.prod(shape)); mat[attn["sparse_indices"]] = vals
            mat = mat.reshape(shape)
            if mat.ndim == 4: token_attn = mat[0].mean(axis=0)
            else: token_attn = mat
            for ti in range(min(len(tokens), token_attn.shape[0])):
                intensity[ti, idx] += np.sum(np.abs(token_attn[ti]))
    for ti in range(len(tokens)):
        if tokens[ti] not in ["[CLS]","[SEP]","<s>","</s>","<pad>"]:
            peaks[tokens[ti]] = int(np.argmax(intensity[ti]))
    return peaks

def calc_window_csrv(logs, patcher, inputs, reasoning_tokens, tokens, shap):
    if not logs or not reasoning_tokens: return 0.0
    peaks = get_token_peak_layers(logs, tokens)
    if not peaks: return 0.0
    verified, total = 0, 0
    for t in reasoning_tokens[:CONFIG["top_k_attributions"]]:
        clean = t.replace("##","").strip()
        peak = next((p for tok,p in peaks.items() if clean in tok or tok in clean), None)
        if peak is None: continue
        total += 1
        if patcher.verify_window(inputs, peak): verified += 1
    return verified/max(total,1)

# ================= RUN & AGGREGATE =================
def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = load_dataset(CONFIG["dataset"], CONFIG["dataset_config"], split="validation")
    texts = ds["sentence"][:CONFIG["num_samples"]]
    results = {}
    
    for m_cfg in CONFIG["models"]:
        model_cls = m_cfg["tokenizer"]
        model = (BertForSequenceClassification if m_cfg["type"]=="bert" else RobertaForSequenceClassification).from_pretrained(m_cfg["name"]).to(CONFIG["device"])
        tokenizer = model_cls.from_pretrained(m_cfg["name"])
        model.eval()
        
        patcher = MultiLayerPatcher(model, tokenizer, CONFIG["device"])
        patcher.collect_baseline(["This is neutral.","The weather is fine.","I am reading."])
        attr_model = attr.InputXGradient(model)
        
        csrvs = []
        for txt in texts[:100]: # Subsample for ablation speed
            inp = tokenizer(txt, return_tensors="pt", truncation=True, max_length=128).to(CONFIG["device"])
            toks = tokenizer.convert_ids_to_tokens(inp["input_ids"][0])
            try: shap = attr_model.attribute(inp["input_ids"], target=1).abs().sum(dim=-1).squeeze().cpu().numpy()
            except: shap = np.zeros(len(toks))
            top_toks = [toks[i] for i in np.argsort(np.abs(shap))[::-1][:CONFIG["top_k_attributions"]] if i<len(toks)]
            
            eng = enable_logging(model, mode=CONFIG["mtrace_mode"])
            with torch.no_grad(): model(**inp)
            logs = eng.collect_logs(); eng.disable_logging()
            csrvs.append(calc_window_csrv(logs, patcher, inp, top_toks, toks, shap))
            
        results[m_cfg["type"]] = {"csrv": float(np.mean(csrvs)), "std": float(np.std(csrvs, ddof=1))}
        
    path = CONFIG["results_dir"] / f"exp2_multilayer_seed{seed}_results.json"
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump({"seed": seed, "metrics": results}, f, indent=2)
    return results

def aggregate():
    seeds = [json.load(open(CONFIG["results_dir"]/f"exp2_multilayer_seed{s}_results.json"))["metrics"] 
             for s in CONFIG["seeds"] if (CONFIG["results_dir"]/f"exp2_multilayer_seed{s}_results.json").exists()]
    if not seeds: return
    agg = {m: {"csrv": [], "std": []} for m in ["bert","roberta"]}
    for s in seeds:
        for m in ["bert","roberta"]:
            if m in s: agg[m]["csrv"].append(s[m]["csrv"]); agg[m]["std"].append(s[m]["std"])
    final = {}
    for m in ["bert","roberta"]:
        if agg[m]["csrv"]:
            final[m] = {"mean": float(np.mean(agg[m]["csrv"])), "std": float(np.std(agg[m]["csrv"], ddof=1))}
    print(f"BERT Multi-Layer CSVR: {final['bert']['mean']:.3f} ± {final['bert']['std']:.3f}")
    print(f"RoBERTa Multi-Layer CSVR: {final['roberta']['mean']:.3f} ± {final['roberta']['std']:.3f}")
    return final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        for s in CONFIG["seeds"]: run_seed(s)
        aggregate()
    elif args.seed: run_seed(args.seed)
    elif args.aggregate: aggregate()
    else: run_seed(42)