import os
import sys
from pathlib import Path
import traceback
import dataclasses

def main():
    print("Running stage smoke test...")
    # Add stage-runtime to PYTHONPATH
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root / "stage-runtime"))
    # Add interceptor-training to PYTHONPATH for src imports
    sys.path.insert(0, str(repo_root / "interceptor-training"))
    
    try:
        from stage_runtime.stage_config import discover_stage_configs, load_stage_config
        from stage_runtime.runner import run_stage
        
        configs_dir = repo_root / "stage-runtime" / "configs" / "stage"
        configs = discover_stage_configs(configs_dir)
        if not configs:
            print("No configs found!")
            sys.exit(1)
            
        print(f"Found {len(configs)} configs.")
        
        for path in configs:
            print(f"Testing {path.name}...")
            cfg = load_stage_config(path)
            
            # Force control to open-loop instructions so we don't need torch
            if cfg.control == "model":
                cfg = dataclasses.replace(cfg, control="open_loop")
                
            cfg = dataclasses.replace(cfg, segments=1, segment_seconds=0.5)
            
            out_dir = repo_root / "stage-runtime" / "out"
            report = run_stage(cfg, out_dir=out_dir, data_roots=[], render=False, write_sidecar=False, verbose=False)
            print(f"  Success: {report['video']['frames']} frames simulated.")
            
        print("Smoke test passed.")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
