#!/usr/bin/env python3
"""
HumanEval Code Generation with Ollama
Part 1: Prompt Design & Code Generation

This script evaluates different prompting strategies on HumanEval problems
using local LLMs via Ollama (no API keys required).
"""

import json
import subprocess
import re
from typing import List, Dict, Any
import os
from collections import defaultdict

# Install required packages
def install_requirements():
    """Install necessary packages"""
    packages = ['human-eval']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run(['pip', 'install', package, '--break-system-packages'], 
                         check=True, capture_output=True)

install_requirements()

from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


class PromptStrategy:
    """Base class for prompting strategies"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        raise NotImplementedError


class DirectPrompt(PromptStrategy):
    """Direct/Baseline prompting - just the problem"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# Complete the function above. Only provide the implementation, no explanations."""


class ChainOfThoughtPrompt(PromptStrategy):
    """Chain-of-Thought (CoT) prompting"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# You are an expert Python programmer with 10+ years of experience in algorithm design and software engineering.
# Your code is known for being correct, efficient, and well-tested.
# 
# As an expert, solve this problem by thinking step by step:
# 1. First, carefully understand what the function needs to do
#    - Read the docstring and examples thoroughly
#    - Identify input types, output types, and constraints
# 2. Think about the approach and algorithm like an expert would
#    - Consider the most appropriate algorithm or data structure
#    - Think about time and space complexity
# 3. Consider edge cases that could break the code
#    - Empty inputs, single elements, boundary conditions
#    - Invalid inputs and how to handle them
# 4. Implement the solution with best practices
#    - Use clear variable names
#    - Initialize all variables before use
#    - Write clean, readable code
#
# As an expert programmer, provide your complete, production-ready implementation:"""


class StepwiseCoTPrompt(PromptStrategy):
    """Stepwise Chain-of-Thought (SCoT) prompting"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# Step 1: Analyze the problem requirements
# Step 2: Identify input/output specifications
# Step 3: Design the algorithm
# Step 4: Handle edge cases
# Step 5: Implement the solution

# Implementation:"""


class SelfPlanningPrompt(PromptStrategy):
    """Self-Planning prompting"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# You are a senior software engineer at a top tech company, known for writing robust, bug-free code.
# Before implementing any solution, you always plan carefully and consider all aspects.
# 
# As a senior engineer, plan your implementation by answering these questions:
# 
# - What are the inputs and expected outputs?
#   (Think like an engineer: be precise about types, formats, and constraints)
# 
# - What algorithm or approach should I use?
#   (Consider: What would be the most elegant and efficient solution?)
# 
# - What are the key steps in my solution?
#   (Break it down: What's the sequence of operations?)
# 
# - What edge cases do I need to handle?
#   (Think defensively: What could go wrong? Empty inputs? Invalid data?)
# 
# - What variables do I need to track?
#   (Be thorough: List all state that needs to be maintained)
# 
# Based on this careful planning, here's my professional implementation:"""


class SelfDebuggingPrompt(PromptStrategy):
    """Self-Debugging prompting"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# Before implementing, let me consider:
# - Common mistakes that could occur
# - How to validate the solution
# - Edge cases that might break the code
# - Type checking and error handling

# Debugged implementation:"""


class SelfEditPrompt(PromptStrategy):
    """Self-Edit prompting"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# First draft considerations:
# - Write a clear, working solution
# - Review for improvements
# - Optimize if possible
# - Ensure readability

# Final edited implementation:"""


class SelfRepairPrompt(PromptStrategy):
    """Self-Repair prompting"""
    
    @staticmethod
    def format_prompt(problem: Dict[str, Any]) -> str:
        return f"""{problem['prompt']}
# Implementation with self-repair checks:
# - Write the solution
# - Mentally test with examples
# - Fix any identified issues
# - Verify correctness

# Repaired implementation:"""


class OllamaEvaluator:
    """Evaluator for LLM code generation using Ollama"""
    
    def __init__(self, models: List[str]):
        """
        Initialize evaluator with list of Ollama models
        
        Args:
            models: List of model names available in Ollama
        """
        self.models = models
        self.strategies = {
            'Direct': DirectPrompt,
            'CoT': ChainOfThoughtPrompt,
            'SCoT': StepwiseCoTPrompt,
            'Self-Planning': SelfPlanningPrompt,
            'Self-Debugging': SelfDebuggingPrompt,
            'Self-Edit': SelfEditPrompt,
            'Self-Repair': SelfRepairPrompt
        }
        
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def generate_code(self, model: str, prompt: str, n_samples: int = 5) -> List[str]:
        """
        Generate code samples using Ollama
        
        Args:
            model: Model name
            prompt: Prompt text
            n_samples: Number of samples to generate
            
        Returns:
            List of generated code samples
        """
        samples = []
        
        for i in range(n_samples):
            try:
                # Call ollama via subprocess
                result = subprocess.run(
                    ['ollama', 'run', model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    generated_text = result.stdout.strip()
                    # Extract only the code part
                    code = self.extract_code(generated_text)
                    samples.append(code)
                    print(f"  Sample {i+1}/{n_samples} generated")
                else:
                    print(f"  Error generating sample {i+1}: {result.stderr}")
                    samples.append("")
                    
            except subprocess.TimeoutExpired:
                print(f"  Timeout for sample {i+1}")
                samples.append("")
            except Exception as e:
                print(f"  Exception for sample {i+1}: {e}")
                samples.append("")
        
        return samples
    
    def extract_code(self, text: str) -> str:
        """
        Extract Python code from generated text
        
        Args:
            text: Generated text containing code
            
        Returns:
            Extracted code
        """
        # Strategy 1: Try to extract code between ```python and ```
        code_block_pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Get the first match and clean it
            code = matches[0].strip()
            # Remove any [TESTS] or assertion blocks
            code = re.sub(r'\[TESTS\].*', '', code, flags=re.DOTALL)
            code = re.sub(r'assert .*', '', code, flags=re.MULTILINE)
            code = re.sub(r'# Test.*', '', code, flags=re.MULTILINE)
            return code.strip()
        
        # Strategy 2: Try to extract code between ``` and ```
        code_block_pattern = r'```\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                # Skip if it contains [PYTHON] tags or test markers
                if '[PYTHON]' not in match and 'assert' not in match[:100]:
                    code = match.strip()
                    # Clean up
                    code = re.sub(r'\[TESTS\].*', '', code, flags=re.DOTALL)
                    return code.strip()
        
        # Strategy 3: Look for [PYTHON] tags
        pattern = r'\[PYTHON\](.*?)(?:\[/PYTHON\]|\[TESTS\]|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            # Remove ``` if present
            code = re.sub(r'^```\w*\s*\n', '', code)
            code = re.sub(r'\n```$', '', code)
            # Remove test cases
            code = re.sub(r'assert .*', '', code, flags=re.MULTILINE)
            code = re.sub(r'# Test.*', '', code, flags=re.MULTILINE)
            return code.strip()
        
        # Strategy 4: Try to find function definition directly
        lines = text.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
            if in_function:
                # Stop at assertions, tests, or special markers
                if any(line.strip().startswith(x) for x in ['assert', '# Test', '```', '[TESTS]', 'if __name__']):
                    break
                code_lines.append(line)
                # Stop at blank line after function (simple heuristic)
                if in_function and line.strip() == '' and code_lines and any('return' in l for l in code_lines):
                    break
        
        if code_lines:
            # Clean up trailing empty lines
            while code_lines and not code_lines[-1].strip():
                code_lines.pop()
            return '\n'.join(code_lines).strip()
        
        # Return the whole text if nothing else works (fallback)
        return text.strip()
    
    def evaluate_on_problems(self, 
                           problem_ids: List[str],
                           strategies: List[str],
                           n_samples: int = 5) -> Dict[str, Any]:
        """
        Evaluate models on selected problems with different strategies
        
        Args:
            problem_ids: List of HumanEval problem IDs
            strategies: List of strategy names to use
            n_samples: Number of samples per problem (for pass@k)
            
        Returns:
            Dictionary with evaluation results
        """
        # Load HumanEval problems
        problems = read_problems()
        
        results = defaultdict(lambda: defaultdict(dict))
        
        # Filter selected problems
        selected_problems = {pid: problems[pid] for pid in problem_ids if pid in problems}
        
        print(f"\nEvaluating {len(selected_problems)} problems with {len(strategies)} strategies on {len(self.models)} models")
        print(f"Generating {n_samples} samples per problem (for pass@{n_samples})")
        print("="*80)
        
        for model in self.models:
            print(f"\nModel: {model}")
            print("-"*80)
            
            for strategy_name in strategies:
                print(f"\n  Strategy: {strategy_name}")
                strategy = self.strategies[strategy_name]
                
                # Prepare samples for evaluation
                samples = []
                
                for task_id, problem in selected_problems.items():
                    print(f"    Problem: {task_id}")
                    
                    # Generate prompt
                    prompt = strategy.format_prompt(problem)
                    
                    # Generate code samples
                    generated_codes = self.generate_code(model, prompt, n_samples)
                    
                    # Create completion entries for evaluation
                    for i, code in enumerate(generated_codes):
                        # Combine prompt with generated code
                        full_code = problem['prompt'] + code
                        
                        samples.append({
                            'task_id': task_id,
                            'completion': full_code
                        })
                
                # Save samples to file
                sample_file = f'{model}_{strategy_name}.jsonl'
                sample_file = sample_file.replace(':', '_').replace('/', '_')
                write_jsonl(sample_file, samples)
                
                # Evaluate using human-eval
                try:
                    print(f"\n    Evaluating {len(samples)} samples...")
                    eval_results = evaluate_functional_correctness(
                        sample_file,
                        k=[1, min(5, n_samples)],
                        n_workers=4,
                        timeout=3.0
                    )
                    
                    results[model][strategy_name] = eval_results
                    print(f"    Results: {eval_results}")
                    
                except Exception as e:
                    print(f"    Evaluation error: {e}")
                    results[model][strategy_name] = {'error': str(e)}
        
        return dict(results)
    
    def print_results_table(self, results: Dict[str, Any]):
        """Print results in a formatted table"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        for model, strategy_results in results.items():
            print(f"\nModel: {model}")
            print("-"*80)
            print(f"{'Strategy':<20} {'pass@1':<10} {'pass@5':<10}")
            print("-"*80)
            
            for strategy, metrics in strategy_results.items():
                if 'error' in metrics:
                    print(f"{strategy:<20} {'ERROR':<10} {'ERROR':<10}")
                else:
                    pass_at_1 = metrics.get('pass@1', 0) * 100
                    pass_at_5 = metrics.get('pass@5', 0) * 100
                    print(f"{strategy:<20} {pass_at_1:<10.2f}% {pass_at_5:<10.2f}%")
            print()


def main():
    """Main execution function"""
    
    print("HumanEval Code Generation Evaluation with Ollama")
    print("="*80)
    
    # Configuration
    MODELS = [
        'codellama:7b',  # CodeLlama family
        'deepseek-coder:6.7b',  # DeepSeek family (different from CodeLlama)
    ]
    
    # Select 10 problems from HumanEval
    SELECTED_PROBLEMS = [
        'HumanEval/0',   # has_close_elements
        'HumanEval/1',   # separate_paren_groups
        'HumanEval/2',   # truncate_number
        'HumanEval/10',  # make_palindrome
        'HumanEval/11',  # string_xor
        'HumanEval/12',  # longest
        'HumanEval/20',  # find_closest_elements
        'HumanEval/25',  # factorize
        'HumanEval/31',  # is_prime
        'HumanEval/37',  # sort_even
    ]
    
    # Select 2 prompting strategies for evaluation
    STRATEGIES = [
        'CoT',
        'Self-Planning'
    ]
    
    # Initialize evaluator
    evaluator = OllamaEvaluator(MODELS)
    
    # Check Ollama installation
    if not evaluator.check_ollama_installed():
        print("\nERROR: Ollama is not installed or not running!")
        print("\nTo install Ollama:")
        print("1. Visit: https://ollama.ai")
        print("2. Download and install for your platform")
        print("3. Run: ollama pull codellama:7b")
        print("4. Run: ollama pull deepseek-coder:6.7b")
        return
    
    print("\n✓ Ollama is installed and running")
    
    # Check if models are available
    print("\nChecking models...")
    for model in MODELS:
        try:
            result = subprocess.run(['ollama', 'show', model], 
                                  capture_output=True, 
                                  timeout=5)
            if result.returncode == 0:
                print(f"  ✓ {model} is available")
            else:
                print(f"  ✗ {model} is NOT available. Run: ollama pull {model}")
        except:
            print(f"  ✗ Error checking {model}")
    
    print("\nStarting evaluation...")
    print(f"Problems: {len(SELECTED_PROBLEMS)}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Models: {MODELS}")
    print(f"Samples per problem: 5 (for pass@5)")
    print(f"Total evaluations: 2 models × 2 strategies × 10 problems × 5 samples = 200 code generations")
    
    # Run evaluation
    results = evaluator.evaluate_on_problems(
        problem_ids=SELECTED_PROBLEMS,
        strategies=STRATEGIES,
        n_samples=5
    )
    
    # Print results table
    evaluator.print_results_table(results)
    
    # Save results to JSON
    results_file = 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
