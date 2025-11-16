"""
Dependency Checker for Text Classification System
Run this to verify all required packages are installed correctly.
"""

import sys

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'kagglehub': 'kagglehub',
        'nltk': 'nltk',
        'wordcloud': 'wordcloud',
        'imblearn': 'imbalanced-learn',
        'PIL': 'Pillow',
        'scipy': 'scipy',
        'joblib': 'joblib'
    }
    
    print("Checking dependencies...\n")
    print("="*60)
    
    missing = []
    installed = []
    
    for module_name, package_name in required_packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            installed.append((package_name, version))
            print(f"✓ {package_name:20s} : {version}")
        except ImportError:
            missing.append(package_name)
            print(f"✗ {package_name:20s} : NOT INSTALLED")
    
    print("="*60)
    print(f"\nInstalled: {len(installed)}/{len(required_packages)}")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        print("\nPython version:", sys.version)
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)