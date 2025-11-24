"""
Robust Test Suite for Lead Scoring System
Tests multiple scenarios including edge cases
"""

from lead_scoring_app import LeadScoringSystem

def test_scoring_system():
    print("="*70)
    print("LEAD SCORING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Initialize scorer (make sure model file exists)
    try:
        scorer = LeadScoringSystem('lead_scoring_model_xgb.pkl')
        print("✅ Model loaded successfully\n")
    except FileNotFoundError:
        print("❌ Error: lead_scoring_model_xgb.pkl not found")
        print("Run: python3 train_model.py first\n")
        return
    
    # Test cases covering various scenarios
    test_leads = [
        {
            'name': '🔥 Ultra High-Value Lead (Wealthy + Ready)',
            'expected': 'High',
            'data': {
                'age': 45,
                'income': 350000,
                'credit_score': 820,
                'website_visits': 35,
                'email_opens': 20,
                'property_views': 22,
                'contact_attempts': 8,
                'time_on_site_minutes': 120,
                'inquiry_submitted': 1,
                'financing_pre_approved': 1,
                'budget_disclosed': 1,
                'location_preference_set': 1
            }
        },
        {
            'name': '✅ High-Value Lead (Standard)',
            'expected': 'High',
            'data': {
                'age': 42,
                'income': 220000,
                'credit_score': 780,
                'website_visits': 25,
                'email_opens': 15,
                'property_views': 18,
                'contact_attempts': 5,
                'time_on_site_minutes': 90,
                'inquiry_submitted': 1,
                'financing_pre_approved': 1,
                'budget_disclosed': 1,
                'location_preference_set': 1
            }
        },
        {
            'name': '🟡 Medium-Value Lead (Good Potential)',
            'expected': 'Medium',
            'data': {
                'age': 35,
                'income': 110000,
                'credit_score': 720,
                'website_visits': 15,
                'email_opens': 8,
                'property_views': 10,
                'contact_attempts': 3,
                'time_on_site_minutes': 50,
                'inquiry_submitted': 1,
                'financing_pre_approved': 0,
                'budget_disclosed': 1,
                'location_preference_set': 1
            }
        },
        {
            'name': '🟡 Medium-Value Lead (Browsing)',
            'expected': 'Medium',
            'data': {
                'age': 38,
                'income': 95000,
                'credit_score': 690,
                'website_visits': 10,
                'email_opens': 6,
                'property_views': 8,
                'contact_attempts': 2,
                'time_on_site_minutes': 40,
                'inquiry_submitted': 0,
                'financing_pre_approved': 1,
                'budget_disclosed': 1,
                'location_preference_set': 1
            }
        },
        {
            'name': '🔴 Low-Value Lead (Young, Low Income)',
            'expected': 'Low',
            'data': {
                'age': 24,
                'income': 42000,
                'credit_score': 590,
                'website_visits': 3,
                'email_opens': 1,
                'property_views': 2,
                'contact_attempts': 0,
                'time_on_site_minutes': 10,
                'inquiry_submitted': 0,
                'financing_pre_approved': 0,
                'budget_disclosed': 0,
                'location_preference_set': 0
            }
        },
        {
            'name': '🔴 Low-Value Lead (Poor Credit)',
            'expected': 'Low',
            'data': {
                'age': 32,
                'income': 65000,
                'credit_score': 520,
                'website_visits': 5,
                'email_opens': 2,
                'property_views': 3,
                'contact_attempts': 1,
                'time_on_site_minutes': 15,
                'inquiry_submitted': 0,
                'financing_pre_approved': 0,
                'budget_disclosed': 0,
                'location_preference_set': 0
            }
        },
        {
            'name': '⚠️  Edge Case: Rich but No Engagement',
            'expected': 'Medium or Low',
            'data': {
                'age': 55,
                'income': 300000,
                'credit_score': 800,
                'website_visits': 1,
                'email_opens': 0,
                'property_views': 1,
                'contact_attempts': 0,
                'time_on_site_minutes': 5,
                'inquiry_submitted': 0,
                'financing_pre_approved': 0,
                'budget_disclosed': 0,
                'location_preference_set': 0
            }
        },
        {
            'name': '⚠️  Edge Case: Engaged but Poor Credit',
            'expected': 'Medium or Low',
            'data': {
                'age': 29,
                'income': 55000,
                'credit_score': 600,
                'website_visits': 30,
                'email_opens': 18,
                'property_views': 20,
                'contact_attempts': 7,
                'time_on_site_minutes': 100,
                'inquiry_submitted': 1,
                'financing_pre_approved': 0,
                'budget_disclosed': 1,
                'location_preference_set': 1
            }
        },
        {
            'name': '⚠️  Edge Case: Borderline Everything',
            'expected': 'Medium',
            'data': {
                'age': 40,
                'income': 90000,
                'credit_score': 680,
                'website_visits': 12,
                'email_opens': 7,
                'property_views': 9,
                'contact_attempts': 2,
                'time_on_site_minutes': 45,
                'inquiry_submitted': 0,
                'financing_pre_approved': 1,
                'budget_disclosed': 0,
                'location_preference_set': 1
            }
        }
    ]
    
    # Run predictions
    correct = 0
    total = len(test_leads)
    
    for i, test_lead in enumerate(test_leads, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{total}: {test_lead['name']}")
        print(f"Expected: {test_lead['expected']}")
        print(f"{'─'*70}")
        
        # Show key features
        data = test_lead['data']
        print(f"  Income: ${data['income']:,} | Credit: {data['credit_score']} | Age: {data['age']}")
        print(f"  Engagement: {data['website_visits']} visits, {data['property_views']} properties, {data['time_on_site_minutes']}min")
        print(f"  Intent: Inquiry={data['inquiry_submitted']}, Financed={data['financing_pre_approved']}, Budget={data['budget_disclosed']}")
        
        # Predict
        result = scorer.predict_single(data)
        
        # Display results
        print(f"\n  🎯 Predicted: {result['lead_score']} (Confidence: {result['confidence']:.1%})")
        print(f"  📊 Probabilities:")
        for class_name, prob in result['probabilities'].items():
            bar = '█' * int(prob * 30)
            print(f"     {class_name:6s}: {prob:.1%} {bar}")
        print(f"  💡 Action: {result['recommendation']}")
        
        # Check if correct
        if 'or' in test_lead['expected']:
            options = test_lead['expected'].split(' or ')
            is_correct = result['lead_score'] in options
        else:
            is_correct = result['lead_score'] == test_lead['expected']
        
        if is_correct:
            correct += 1
            print(f"  ✅ PASS")
        else:
            print(f"  ⚠️  Unexpected (got {result['lead_score']}, expected {test_lead['expected']})")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Passed: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_scoring_system()
