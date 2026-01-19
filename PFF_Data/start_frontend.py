#!/usr/bin/env python3
"""
QB Archetype Analysis Frontend Startup Script
Launches the Flask API with the web frontend
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🏈 QB Archetype Analysis Frontend")
    print("=" * 50)
    
    # Check if enhanced_api.py exists
    if not Path("enhanced_api.py").exists():
        print("❌ Error: enhanced_api.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if templates directory exists
    if not Path("templates").exists():
        print("❌ Error: templates directory not found!")
        print("Please ensure the frontend templates are in place.")
        sys.exit(1)
    
    print("✅ Starting Flask API with frontend...")
    print("🌐 Frontend will be available at: http://localhost:5001/frontend")
    print("📚 API Documentation: ahttp://localhost:5001/swagger")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Start the Flask API
        subprocess.run([sys.executable, "enhanced_api.py"])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
