#!/usr/bin/env python3
# Quick test script for Phase 2 components

import sys
sys.path.append("src")

def test_imports():
    """Test if all Phase 2 components can be imported"""
    try:
        print("🧪 Testing imports...")
        
        # Test Phase 2 imports
        from rag_engine.vectorstore.faiss_vector_manager import FAISSVectorManager
        print("✅ FAISS Vector Manager import successful")
        
        from response_generation.text_response_generator import TextResponseGenerator
        print("✅ Text Response Generator import successful")
        
        print("\n🎉 All Phase 2 components imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("📋 Please check that all dependencies are installed:")
        print("   pip install -r requirements_phase2.txt")
        return False

if __name__ == "__main__":
    test_imports()
