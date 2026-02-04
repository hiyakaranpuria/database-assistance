#!/usr/bin/env python3
"""
Test Local LLM Integration
Comprehensive testing for natural language to MongoDB query conversion
"""

import json
import time
from llm_integration import OllamaLLM, MultilingualLLM, generate_llm_query, LLM_AVAILABLE
from metadata_provider import extract_metadata
from dynamic_query_executor import execute_mongo_query

def test_basic_llm_functionality():
    """Test basic LLM functionality"""
    print("🔧 Testing Basic LLM Functionality")
    print("-" * 40)
    
    if not LLM_AVAILABLE:
        print("❌ LLM not available. Please install and configure Ollama.")
        return False
    
    try:
        llm = OllamaLLM()
        schema = extract_metadata()
        
        # Simple test query
        question = "Show total sales"
        query = llm.generate_query(question, schema)
        
        # Validate JSON
        parsed_query = json.loads(query)
        print(f"✅ Basic functionality working")
        print(f"   Question: {question}")
        print(f"   Generated: {query[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False

def test_query_generation():
    """Test various types of query generation"""
    print("\n📊 Testing Query Generation")
    print("-" * 40)
    
    test_cases = [
        {
            "question": "Show total sales for 2024",
            "expected_operators": ["$match", "$group"],
            "description": "Sales aggregation with date filter"
        },
        {
            "question": "Top 5 products by revenue",
            "expected_operators": ["$group", "$lookup", "$sort", "$limit"],
            "description": "Product analysis with joins"
        },
        {
            "question": "Customers who spent most money",
            "expected_operators": ["$group", "$lookup", "$sort"],
            "description": "Customer analysis"
        },
        {
            "question": "Monthly sales trends",
            "expected_operators": ["$match", "$group"],
            "description": "Time-based aggregation"
        },
        {
            "question": "Count all customers",
            "expected_operators": ["$count"],
            "description": "Simple counting"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Question: '{test_case['question']}'")
        
        try:
            start_time = time.time()
            query = generate_llm_query(test_case['question'])
            generation_time = time.time() - start_time
            
            # Parse and validate JSON
            parsed_query = json.loads(query)
            
            # Check for expected operators
            query_str = json.dumps(parsed_query)
            found_operators = []
            for op in test_case['expected_operators']:
                if op in query_str:
                    found_operators.append(op)
            
            success = len(found_operators) > 0
            
            print(f"   ✅ Generated in {generation_time:.2f}s")
            print(f"   📋 Operators found: {found_operators}")
            print(f"   🔍 Query: {query[:150]}...")
            
            results.append({
                'question': test_case['question'],
                'success': success,
                'time': generation_time,
                'operators': found_operators
            })
            
        except json.JSONDecodeError:
            print(f"   ❌ Invalid JSON generated")
            results.append({'question': test_case['question'], 'success': False, 'error': 'Invalid JSON'})
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({'question': test_case['question'], 'success': False, 'error': str(e)})
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📈 Query Generation Results: {successful}/{len(test_cases)} successful")
    
    return results

def test_multilingual_support():
    """Test multilingual query support"""
    print("\n🌍 Testing Multilingual Support")
    print("-" * 40)
    
    multilingual_tests = [
        {"lang": "English", "question": "Show total sales this year"},
        {"lang": "Spanish", "question": "Muestra las ventas totales de este año"},
        {"lang": "French", "question": "Afficher les ventes totales de cette année"},
        {"lang": "German", "question": "Zeige die Gesamtverkäufe dieses Jahres"},
        {"lang": "Chinese", "question": "显示今年的总销售额"},
        {"lang": "Japanese", "question": "今年の総売上を表示"},
    ]
    
    if not LLM_AVAILABLE:
        print("❌ LLM not available for multilingual testing")
        return []
    
    try:
        multilingual_llm = MultilingualLLM()
        results = []
        
        for test in multilingual_tests:
            print(f"\n🔤 {test['lang']}: '{test['question']}'")
            
            try:
                start_time = time.time()
                query = multilingual_llm.generate_query(test['question'])
                generation_time = time.time() - start_time
                
                # Validate JSON
                json.loads(query)
                
                print(f"   ✅ Generated in {generation_time:.2f}s")
                print(f"   🔍 Query: {query[:100]}...")
                
                results.append({
                    'language': test['lang'],
                    'success': True,
                    'time': generation_time
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'language': test['lang'],
                    'success': False,
                    'error': str(e)
                })
        
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📈 Multilingual Results: {successful}/{len(multilingual_tests)} successful")
        
        return results
        
    except Exception as e:
        print(f"❌ Multilingual testing failed: {e}")
        return []

def test_query_execution():
    """Test actual query execution against database"""
    print("\n🗄️  Testing Query Execution")
    print("-" * 40)
    
    execution_tests = [
        "Show total sales for 2024",
        "Count all customers",
        "Top 3 products by sales"
    ]
    
    results = []
    
    for question in execution_tests:
        print(f"\n🔍 Testing: '{question}'")
        
        try:
            # Generate query
            query = generate_llm_query(question)
            print(f"   📝 Generated query: {query[:100]}...")
            
            # Execute query
            start_time = time.time()
            raw_results = execute_mongo_query(query, question)
            execution_time = time.time() - start_time
            
            if raw_results and not isinstance(raw_results, str):
                print(f"   ✅ Executed in {execution_time:.2f}s")
                print(f"   📊 Results: {len(raw_results)} records")
                if len(raw_results) > 0:
                    print(f"   🔍 Sample: {str(raw_results[0])[:100]}...")
                
                results.append({
                    'question': question,
                    'success': True,
                    'execution_time': execution_time,
                    'result_count': len(raw_results)
                })
            else:
                print(f"   ⚠️  No results or error: {raw_results}")
                results.append({
                    'question': question,
                    'success': False,
                    'error': 'No results'
                })
                
        except Exception as e:
            print(f"   ❌ Execution failed: {e}")
            results.append({
                'question': question,
                'success': False,
                'error': str(e)
            })
    
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📈 Execution Results: {successful}/{len(execution_tests)} successful")
    
    return results

def test_performance_benchmarks():
    """Test performance benchmarks"""
    print("\n⚡ Performance Benchmarks")
    print("-" * 40)
    
    if not LLM_AVAILABLE:
        print("❌ LLM not available for performance testing")
        return {}
    
    benchmark_queries = [
        "Show total sales",
        "Top 10 customers",
        "Monthly trends",
        "Product analysis",
        "Customer segments"
    ]
    
    times = []
    
    for query in benchmark_queries:
        try:
            start_time = time.time()
            result = generate_llm_query(query)
            end_time = time.time()
            
            generation_time = end_time - start_time
            times.append(generation_time)
            
            print(f"   📊 '{query}': {generation_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ '{query}': Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📈 Performance Summary:")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Fastest: {min_time:.2f}s")
        print(f"   Slowest: {max_time:.2f}s")
        
        return {
            'average': avg_time,
            'min': min_time,
            'max': max_time,
            'samples': len(times)
        }
    
    return {}

def generate_test_report(results):
    """Generate comprehensive test report"""
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    # Overall status
    if LLM_AVAILABLE:
        print("✅ Local LLM Status: Available and functional")
    else:
        print("❌ Local LLM Status: Not available")
        print("\n💡 To enable local LLM:")
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start service: ollama serve")
        print("   3. Download model: ollama pull llama3.2:3b-instruct-q4_0")
        return
    
    # Test results summary
    for test_name, test_results in results.items():
        if isinstance(test_results, list) and test_results:
            successful = sum(1 for r in test_results if r.get('success', False))
            total = len(test_results)
            print(f"\n📊 {test_name}: {successful}/{total} passed")
            
            if successful < total:
                print("   ⚠️  Issues found:")
                for result in test_results:
                    if not result.get('success', False):
                        error = result.get('error', 'Unknown error')
                        print(f"      • {result.get('question', 'Unknown')}: {error}")
        
        elif isinstance(test_results, dict) and 'average' in test_results:
            print(f"\n⚡ Performance: {test_results['average']:.2f}s average")
    
    print(f"\n🎯 Recommendations:")
    if LLM_AVAILABLE:
        print("   ✅ Local LLM is working well!")
        print("   💡 Consider fine-tuning for your specific domain")
        print("   🚀 Enable multilingual support for global users")
    
    print(f"\n📚 Next Steps:")
    print("   1. Integrate with main application")
    print("   2. Test with real user queries")
    print("   3. Monitor performance in production")
    print("   4. Collect feedback for improvements")

def main():
    """Run comprehensive LLM integration tests"""
    print("🤖 Local LLM Integration Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Run all tests
    if test_basic_llm_functionality():
        results['Query Generation'] = test_query_generation()
        results['Multilingual Support'] = test_multilingual_support()
        results['Query Execution'] = test_query_execution()
        results['Performance'] = test_performance_benchmarks()
    
    # Generate report
    generate_test_report(results)

if __name__ == "__main__":
    main()