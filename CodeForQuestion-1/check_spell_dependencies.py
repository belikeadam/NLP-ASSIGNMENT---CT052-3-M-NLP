"""
Dependency Checker for Spelling Correction System
Run this to verify all required packages are installed correctly.
"""

import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies for Spelling Correction System...\n")
    print("="*60)
    
    required = []
    optional = []
    missing = []
    
    # Check NLTK (required)
    try:
        import nltk
        version = nltk.__version__
        required.append(('nltk', version))
        print(f"✓ nltk               : {version} (REQUIRED)")
    except ImportError:
        missing.append('nltk')
        print(f"✗ nltk               : NOT INSTALLED (REQUIRED)")
    
    # Check tkinter (required - usually pre-installed)
    try:
        import tkinter
        version = tkinter.TkVersion
        required.append(('tkinter', str(version)))
        print(f"✓ tkinter            : {version} (REQUIRED)")
    except ImportError:
        missing.append('tkinter')
        print(f"✗ tkinter            : NOT INSTALLED (REQUIRED)")
        print("  Install: sudo apt-get install python3-tk (Linux)")
        print("  Or reinstall Python with tcl/tk support")
    
    # Check kagglehub (optional)
    try:
        import kagglehub
        version = getattr(kagglehub, '__version__', 'unknown')
        optional.append(('kagglehub', version))
        print(f"✓ kagglehub          : {version} (OPTIONAL)")
    except ImportError:
        print(f"  kagglehub          : NOT INSTALLED (OPTIONAL)")
        print("  Note: Only needed for downloading Kaggle corpus")
    
    # Check standard library modules
    print("\nStandard Library Modules (included with Python):")
    stdlib_modules = [
        'pickle', 're', 'json', 'collections', 'typing', 
        'dataclasses', 'abc', 'threading', 'os', 'time'
    ]
    
    for module_name in stdlib_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:20s} : Available")
        except ImportError:
            print(f"✗ {module_name:20s} : Missing (should be in Python)")
    
    print("="*60)
    print(f"\nRequired packages installed: {len(required)}/2")
    print(f"Optional packages installed: {len(optional)}/1")
    
    if missing:
        print(f"\n❌ Missing REQUIRED packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All required dependencies are installed!")
        print("\nPython version:", sys.version)
        print("\nYou can now run: python spelling_correction.py")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)