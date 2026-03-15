import ast
import random
import string
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Configure logging for Ubuntu terminal output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticProgramGenerator:
    """
    Generates synthetic Python programs with mathematically verifiable execution traces.
    
    Ground Truth Logic (per M-TRACE Plan v3, Exp 1):
    - Variable Binding (ast.Assign) -> Expected in Layers 1-4 (Early)
    - Arithmetic Computation (ast.BinOp) -> Expected in Layers 5-8 (Middle)
    - Output Generation (ast.Call/print) -> Expected in Layers 9-12 (Late)
    """
    
    def __init__(self, num_layers: int = 12):
        self.num_layers = num_layers
        self.variables = list(string.ascii_lowercase)
        
        # Define the Ground Truth Layer Ranges
        self.ground_truth_map = {
            'bind': (1, 4),      # Early layers for state initialization
            'compute': (5, 8),   # Middle layers for transformation
            'output': (9, 12)    # Late layers for projection/generation
        }

    def generate_single_program(self) -> Dict[str, Any]:
        """Generates one synthetic program and its ground truth trace."""
        
        # 1. Construct a simple deterministic program structure
        # Pattern: x = <int>; y = x + <int>; print(y)
        var_x = random.choice([v for v in self.variables if v != 'y'])
        var_y = 'y'
        val1 = random.randint(1, 50)
        val2 = random.randint(1, 50)
        
        code_str = f"{var_x} = {val1}\n{var_y} = {var_x} + {val2}\nprint({var_y})"
        
        try:
            tree = ast.parse(code_str)
            trace = self._extract_ground_truth_trace(tree)
            
            return {
                "code": code_str,
                "valid": True,
                "ground_truth_trace": trace,
                "expected_layer_ranges": self.ground_truth_map.copy()
            }
        except Exception as e:
            logger.error(f"Failed to parse generated code: {e}")
            return {"code": code_str, "valid": False, "error": str(e)}

    def _extract_ground_truth_trace(self, tree: ast.AST) -> List[Dict]:
        """
        Parses the AST to create the 'Oracle' trace.
        This defines exactly WHAT should happen and WHERE (in which layer range) it should happen.
        """
        trace = []
        
        # Walk the AST to find logical steps
        for node in ast.walk(tree):
            step_info = None
            
            # Case 1: Variable Binding
            if isinstance(node, ast.Assign):
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    step_info = {
                        "step_id": f"bind_{target.id}",
                        "step_type": "bind",
                        "description": f"Bind variable '{target.id}'",
                        "ast_node_type": "Assign"
                    }
            
            # Case 2: Arithmetic Computation
            elif isinstance(node, ast.BinOp):
                op_name = type(node.op).__name__
                step_info = {
                    "step_id": f"compute_{op_name}",
                    "step_type": "compute",
                    "description": f"Perform arithmetic ({op_name})",
                    "ast_node_type": "BinOp"
                }
                
            # Case 3: Output Generation
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    step_info = {
                        "step_id": "output_print",
                        "step_type": "output",
                        "description": "Generate final output",
                        "ast_node_type": "Call"
                    }
            
            if step_info:
                # Attach the Ground Truth Layer Range
                step_type = step_info["step_type"]
                step_info["expected_layer_range"] = self.ground_truth_map[step_type]
                trace.append(step_info)
                
        # Sort trace by logical execution order (AST walk order is generally reliable for this simple syntax)
        # In a full compiler, we'd use control flow graphs, but for this synthetic subset, AST walk suffices.
        return trace

    def generate_dataset(self, n_samples: int, output_path: str) -> str:
        """Generates the full dataset and saves it as a pickle file."""
        logger.info(f"Generating dataset of {n_samples} synthetic programs...")
        
        dataset = []
        valid_count = 0
        
        for i in range(n_samples):
            prog = self.generate_single_program()
            if prog.get("valid"):
                dataset.append(prog)
                valid_count += 1
            else:
                logger.warning(f"Sample {i} invalid, skipping.")
        
        # Ensure directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Parquet-ready Pickle (we will convert to Parquet logs later during inference)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
            
        logger.info(f"✅ Dataset generation complete.")
        logger.info(f"   - Total Valid Samples: {valid_count}/{n_samples}")
        logger.info(f"   - Saved to: {output_file.resolve()}")
        logger.info(f"   - Ground Truth Logic: Bind(L1-4), Compute(L5-8), Output(L9-12)")
        
        return str(output_file.resolve())

if __name__ == "__main__":
    # Configuration
    NUM_SAMPLES = 1000  # Sufficient for statistical significance (Plan v3)
    OUTPUT_FILE = "t_trace/experiments/phase2/exp1/data/synthetic_programs_gt.pkl"
    
    generator = SyntheticProgramGenerator(num_layers=12)
    generator.generate_dataset(NUM_SAMPLES, OUTPUT_FILE)