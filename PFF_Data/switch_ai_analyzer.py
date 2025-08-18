#!/usr/bin/env python3
"""
Simple script to switch between AI analyzers
"""

import sys
import re

def switch_analyzer(analyzer_type):
    """Switch the AI analyzer in config.py"""
    if analyzer_type not in ['lightweight', 'qwen']:
        print(f"❌ Invalid analyzer type: {analyzer_type}")
        print("Valid options: 'lightweight' or 'qwen'")
        return False
    
    # Read config file
    with open('config.py', 'r') as f:
        content = f.read()
    
    # Replace the AI_ANALYZER_TYPE line
    pattern = r'AI_ANALYZER_TYPE = "[^"]*"'
    replacement = f'AI_ANALYZER_TYPE = "{analyzer_type}"'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # Write back to file
        with open('config.py', 'w') as f:
            f.write(new_content)
        
        print(f"✅ Switched to {analyzer_type} analyzer")
        print(f"🔄 Please restart the API for changes to take effect")
        return True
    else:
        print("❌ Could not find AI_ANALYZER_TYPE in config.py")
        return False

def show_current():
    """Show current analyzer configuration"""
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        
        match = re.search(r'AI_ANALYZER_TYPE = "([^"]*)"', content)
        if match:
            current = match.group(1)
            print(f"🔍 Current AI Analyzer: {current}")
            
            if current == 'lightweight':
                print("   ⚡ Fast, rule-based analysis (instant)")
            else:
                print("   🧠 Detailed, LLM-based analysis (60-120s)")
        else:
            print("❌ Could not determine current analyzer")
    except FileNotFoundError:
        print("❌ config.py not found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🤖 AI Analyzer Switcher")
        print("Usage:")
        print("  python3 switch_ai_analyzer.py lightweight  # Fast analysis")
        print("  python3 switch_ai_analyzer.py qwen         # Detailed analysis")
        print("  python3 switch_ai_analyzer.py status       # Show current")
        show_current()
    elif sys.argv[1] == 'status':
        show_current()
    else:
        switch_analyzer(sys.argv[1])
