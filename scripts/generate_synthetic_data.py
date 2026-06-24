#!/usr/bin/env python
"""
Synthetic trajectory generator for opencode-sft dataset augmentation.
Generates conversations where the assistant uses its tool interface (bash, glob, grep, etc.)
to help users with software engineering tasks.
"""

import os
import json
import random
import time
from typing import Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def load_dataset(split="train", num_samples=10) -> List[Dict]:
    """Load samples from HuggingFace dataset."""
    from datasets import load_dataset
    dataset = load_dataset("nkasmanoff/opencode-sft", split=split)
    # Shuffle and take num_samples
    idx = random.sample(range(len(dataset)), num_samples)
    selected = dataset.select(idx)
    return list(selected)

def format_example_for_prompt(sample: Dict) -> str:
    """Format a dataset sample into a readable example for the prompt."""
    messages = sample['messages']
    lines = []
    
    # System prompt
    system_content = messages[0]['content']
    lines.append("System: " + system_content[:200] + "...")
    
    # Conversation
    for msg in messages[1:]:
        role = msg['role']
        if role == 'user':
            lines.append(f"\nUser: {msg['content']}")
        elif role == 'assistant':
            content = msg.get('content', '')
            tool_calls = msg.get('tool_calls', [])
            reasoning = msg.get('reasoning_content', '')
            
            lines.append(f"\nAssistant: {content}")
            if reasoning and reasoning != content:
                lines.append(f"  [Thinking]: {reasoning[:200]}...")
            for tc in tool_calls:
                func = tc.get('function', {})
                name = func.get('name', 'unknown')
                args = json.loads(func.get('arguments', '{}'))
                lines.append(f"  Tool: {name}(args={json.dumps(args, indent=4)[:200]})")
        elif role == 'tool':
            tool = msg.get('tool', 'unknown')
            result = msg.get('content', '')[:100] + "..."
            lines.append(f"\n[Tool Output - {tool}]: {result}")
    
    return "\n".join(lines)

def create_generation_prompt(examples: List[Dict]) -> str:
    """Create the prompt for generating new synthetic trajectories."""
    
    examples_formatted = []
    for i, ex in enumerate(examples, 1):
        examples_formatted.append(format_example_for_prompt(ex))
    
    examples_str = "\n---\n\n".join(examples_formatted)
    
    # Build the JSON structure examples as strings to avoid f-string nesting issues
    sample_structure = json.dumps({
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "reasoning_content": "..."},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls -la\"}"}}]},
            {"role": "tool", "tool": "bash", "content": "..."}
        ]
    })
    
    output_structure = json.dumps({
        "messages": [
            {"role": "system", "content": "You are opencode..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "reasoning_content": "..."},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "tool_name", "arguments": "{\"arg\": \"value\"}"}}]},
            {"role": "tool", "tool": "tool_name", "content": "..."}
        ]
    })
    
    return """You are an expert at creating synthetic training data for AI coding assistants. Your task is to generate NEW, realistic tool-use conversations that the assistant could have had with a user.

The assistant has access to these tools:
- bash: Execute shell commands
- glob: Find files using pattern matching
- grep: Search file contents
- read: Read file contents
- write: Write/create files
- edit: Edit files
- task: Spawn sub-task agents
- webfetch: Fetch web pages/URLs

IMPORTANT REQUIREMENTS:
1. The conversation must follow the SAME FORMAT as the examples below
2. Each assistant turn can use ONE tool call in tool_calls field
3. Assistant should show reasoning (in reasoning_content) before calling tools
4. Tool calls must use valid arguments matching the tool's schema
5. Generate completely different scenarios than the examples - be creative but realistic

FORMAT SPECIFICATIONS:
- Each turn follows this exact structure:
{sample_structure}

- reasoning_content can appear alongside content or separately
- tool_calls is a list with objects containing type, function.name, function.arguments
- tool output messages have role "tool" with tool name and content

TOPICS TO COVER (be diverse):
- File system operations (navigating, creating, editing)
- Code analysis (searching patterns, understanding structure)
- Debugging issues (running commands, checking logs)
- Research tasks (find files, read documentation)
- Multi-step tasks requiring multiple tool calls
- Error cases and recovery

EXAMPLES (Study these patterns carefully):

{examples_str}

---

Now generate a COMPLETELY NEW synthetic conversation. Create a scenario that hasn't appeared in the examples. Make it realistic and follow the same format exactly.

Return ONLY valid JSON with this structure:
{output_structure}
""".format(sample_structure=sample_structure, output_structure=output_structure, examples_str=examples_str)

def generate_single_trajectory(client: OpenAI, examples: List[Dict]) -> Dict:
    """Generate a single synthetic trajectory."""
    prompt = create_generation_prompt(examples)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="z-ai/glm-5.2",
                messages=[
                    {"role": "system", "content": "You are a data generation assistant. You create realistic, synthetic multi-turn conversations for AI coding assistants. Return ONLY valid JSON, no extra text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=3000,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": True}}
            )
            
            raw = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from code blocks
                if '```json' in raw:
                    raw = raw.split('```json')[1].split('```')[0]
                elif '```' in raw:
                    raw = raw.split('```')[1].split('```')[0]
                result = json.loads(raw)
            
            # Validate structure
            if 'messages' not in result or not isinstance(result['messages'], list):
                print(f"  ⚠️ Invalid structure (attempt {attempt+1})")
                continue
            
            # Check for required fields
            has_system = any(m.get('role') == 'system' for m in result['messages'])
            has_user = any(m.get('role') == 'user' for m in result['messages'])
            has_assistant = any(m.get('role') == 'assistant' for m in result['messages'])
            has_tools = any(m.get('role') == 'tool' for m in result['messages'])
            
            if has_system and has_user and has_assistant and has_tools:
                result['source'] = 'synthetic'
                return result
            
            print(f"  ⚠️ Missing required roles in attempt {attempt+1}")
            
        except Exception as e:
            print(f"  ⚠️ API error on attempt {attempt+1}: {e}")
            time.sleep(2)
    
    return None

def validate_trajectory(messages: List[Dict], max_length=50) -> bool:
    """Validate that a trajectory is well-formed."""
    if len(messages) < 3:
        return False
    
    if len(messages) > max_length:
        return False
    
    # Check first message is system
    if messages[0].get('role') != 'system':
        return False
    
    # Check second message is user
    if len(messages) < 2 or messages[1].get('role') != 'user':
        return False
    
    # Check there's at least one assistant turn
    assistant_turns = [m for m in messages if m.get('role') == 'assistant']
    if not assistant_turns:
        return False
    
    # Check tool calls have proper structure
    for msg in messages:
        if 'tool_calls' in msg:
            if not isinstance(msg['tool_calls'], list):
                return False
            for tc in msg['tool_calls']:
                if 'function' not in tc or 'name' not in tc['function']:
                    return False
    
    # Check tool outputs reference actual tool calls
    tool_names_in_calls = set()
    for msg in messages:
        if 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                func_name = tc.get('function', {}).get('name')
                if func_name:
                    tool_names_in_calls.add(func_name)
    
    for msg in messages:
        if msg.get('role') == 'tool':
            if msg.get('tool') not in tool_names_in_calls:
                # This might be ok if it's a generic tool message
                pass
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data for opencode-sft')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of examples to sample from dataset')
    parser.add_argument('--num-to-generate', type=int, default=20, help='Number of synthetic samples to generate')
    parser.add_argument('--output-file', type=str, default='synthetic_conversations.json', help='Output JSON file')
    args = parser.parse_args()
    
    # Set up OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    if not client.api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        sys.exit(1)
    
    # Load dataset
    print("📚 Loading dataset...")
    examples = load_dataset(split='train', num_samples=args.num_samples)
    print(f"✅ Loaded {len(examples)} example trajectories")
    
    # Generate synthetic data
    print(f"\n🤖 Generating {args.num_to_generate} synthetic trajectories...\n")
    
    generated = []
    failed = 0
    
    for i in range(args.num_to_generate):
        print(f"({i+1}/{args.num_to_generate}) Generating...")
        
        # Pick a random example to condition on
        example = random.choice(examples)
        
        result = generate_single_trajectory(client, [example])
        
        if result and validate_trajectory(result['messages']):
            generated.append(result)
            print(f"  ✅ Success (total: {len(generated)})")
        else:
            failed += 1
            print(f"  ❌ Failed (attempted {failed} times)")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    print(f"\n📊 Summary:")
    print(f"  Generated: {len(generated)}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {len(generated)/(len(generated)+failed)*100:.1f}%")
    
    if generated:
        # Save to file
        with open(args.output_file, 'w') as f:
            json.dump(generated, f, indent=2)
        
        # Show a sample
        print(f"\n📝 Sample generated conversation:")
        sample = generated[0]
        print(json.dumps(sample, indent=2)[:1000])
        print("...")
        
        print(f"\n💾 Saved {len(generated)} trajectories to {args.output_file}")

if __name__ == "__main__":
    main()