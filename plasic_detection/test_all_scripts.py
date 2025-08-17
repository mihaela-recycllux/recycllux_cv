#!/usr/bin/env python3
"""
Test script to verify all three plastic detection scripts are working correctly

This script tests the imports and basic functionality of all three detection scripts
without running the full analysis (which requires SentinelHub credentials and 
significant processing time).

Usage:
    python test_all_scripts.py
"""

import sys
import os
import traceback

# Add the plasic_detection directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_script_imports(script_name):
    """Test if a script can be imported successfully"""
    try:
        # Remove .py extension for import
        module_name = script_name.replace('.py', '')
        
        # Try to import the module
        module = __import__(module_name)
        
        print(f"✓ {script_name}: Import successful")
        
        # Check for key functions
        key_functions = []
        
        if 'sentinel2_plastic_fdi' in script_name:
            key_functions = ['setup_credentials', 'calculate_fdi', 'get_sentinel2_data', 
                           'detect_plastic_debris', 'calculate_plastic_area_statistics']
        elif 'sentinel_data_fusion' in script_name:
            key_functions = ['setup_credentials', 'get_sentinel1_data', 'get_sentinel2_data',
                           'create_enhanced_plastic_detection', 'calculate_plastic_area_statistics']
        elif 'comprehensive_plastic' in script_name:
            key_functions = ['setup_credentials', 'calculate_comprehensive_indices', 
                           'detect_plastic_fdi_method', 'detect_plastic_ml_clustering',
                           'create_ensemble_detection', 'calculate_plastic_area_statistics']
        
        missing_functions = []
        for func in key_functions:
            if not hasattr(module, func):
                missing_functions.append(func)
        
        if missing_functions:
            print(f"  ⚠️  Missing functions: {', '.join(missing_functions)}")
        else:
            print(f"  ✓ All key functions present")
            
        return True, None
        
    except Exception as e:
        print(f"❌ {script_name}: Import failed")
        print(f"   Error: {str(e)}")
        return False, str(e)

def test_dependencies():
    """Test if all required dependencies are available"""
    required_packages = [
        'numpy', 'matplotlib', 'sentinelhub', 'sklearn', 
        'dotenv', 'os', 'datetime'
    ]
    
    print("Testing dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('python_dotenv')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_data_directory():
    """Check if the data directory exists and is writable"""
    data_dir = "/Users/varunburde/projects/Recyllux/plasic_detection/data"
    
    print(f"Checking data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir, exist_ok=True)
            print(f"  ✓ Created data directory")
        except Exception as e:
            print(f"  ❌ Cannot create data directory: {e}")
            return False
    else:
        print(f"  ✓ Data directory exists")
    
    # Test write permissions
    test_file = os.path.join(data_dir, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"  ✓ Write permissions OK")
        return True
    except Exception as e:
        print(f"  ❌ Write permission error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 80)
    print("PLASTIC DETECTION SCRIPTS - SYSTEM TEST")
    print("=" * 80)
    
    # Test 1: Dependencies
    print("\n1. Testing Dependencies...")
    deps_ok, missing = test_dependencies()
    if not deps_ok:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Please install with: pip install -r requirements.txt")
        return False
    
    # Test 2: Data directory
    print("\n2. Testing Data Directory...")
    data_dir_ok = check_data_directory()
    if not data_dir_ok:
        print("❌ Data directory setup failed")
        return False
    
    # Test 3: Script imports
    print("\n3. Testing Script Imports...")
    scripts_to_test = [
        '01_sentinel2_plastic_fdi_detection.py',
        '02_sentinel_data_fusion.py', 
        '03_comprehensive_plastic_detection.py'
    ]
    
    all_imports_ok = True
    for script in scripts_to_test:
        script_path = os.path.join(current_dir, script)
        if os.path.exists(script_path):
            success, error = test_script_imports(script)
            if not success:
                all_imports_ok = False
        else:
            print(f"❌ {script}: File not found")
            all_imports_ok = False
    
    # Test 4: Environment file check
    print("\n4. Checking Environment Configuration...")
    env_file = os.path.join(current_dir, '.env')
    if os.path.exists(env_file):
        print(f"  ✓ .env file found")
        # Check if it has the required variables (without reading actual values)
        with open(env_file, 'r') as f:
            content = f.read()
            required_vars = ['SH_CLIENT_ID', 'SH_CLIENT_SECRET']
            missing_vars = [var for var in required_vars if var not in content]
            if missing_vars:
                print(f"  ⚠️  Missing environment variables: {', '.join(missing_vars)}")
            else:
                print(f"  ✓ Required environment variables present")
    else:
        print(f"  ⚠️  .env file not found")
        print(f"     You'll need to create one with SentinelHub credentials before running scripts")
    
    # Final result
    print("\n" + "=" * 80)
    if all_imports_ok and deps_ok and data_dir_ok:
        print("✅ ALL TESTS PASSED - Scripts are ready to run!")
        print("\nTo run the scripts:")
        print("1. Set up your SentinelHub credentials in .env file")
        print("2. Run any of the three detection scripts:")
        print("   - python 01_sentinel2_plastic_fdi_detection.py")
        print("   - python 02_sentinel_data_fusion.py") 
        print("   - python 03_comprehensive_plastic_detection.py")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above before running scripts")
    print("=" * 80)
    
    return all_imports_ok and deps_ok and data_dir_ok

if __name__ == "__main__":
    main()
