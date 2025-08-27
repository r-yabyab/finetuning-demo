# Java Code Generation Prompts for Pass@K Testing

This document contains various Java class prompts designed to test different programming concepts and complexity levels for LLM code generation evaluation using pass@k metrics.

## Difficulty Levels

### Beginner Level
1. **Calculator** (existing) - Basic arithmetic operations
2. **StringUtils** - String manipulation and basic algorithms

### Intermediate Level  
3. **ArrayProcessor** - Array operations, exception handling, algorithm implementation
4. **DateValidator** - Date logic, leap year calculations, validation
5. **NumberConverter** - Base conversion, prime numbers, input validation
6. **TextAnalyzer** - Text processing, regex-like operations, formatting

### Advanced Level
7. **MathUtils** - Mathematical algorithms, recursion, edge case handling
8. **PasswordValidator** - Complex validation rules, scoring algorithms, pattern detection

## Key Testing Aspects

### Algorithm Complexity
- **Simple**: Basic arithmetic, string reversal
- **Medium**: Array processing, date calculations, base conversion
- **Complex**: Mathematical sequences, text analysis, validation scoring

### Error Handling
- Null input handling
- Invalid input validation  
- Exception throwing and types
- Edge case management

### Data Structure Knowledge
- Array manipulation
- String processing
- Primitive type conversions
- Collection-like operations

### Mathematical Concepts
- Prime number detection
- Factorial and Fibonacci sequences
- GCD/LCM algorithms
- Date arithmetic

### Real-World Applications
- Password security validation
- Text analysis and formatting
- Number system conversions
- Data validation

## Usage Instructions

1. Choose a prompt based on desired difficulty level
2. Use the prompt to generate Java code via LLM
3. Test the generated code against the corresponding JUnit test file
4. Measure pass@k success rates across multiple generations

## Benefits for Pass@K Testing

- **Variety**: Different programming domains and concepts
- **Scalability**: Easy to adjust difficulty by modifying requirements
- **Comprehensive**: Tests multiple aspects of programming ability
- **Realistic**: Based on common programming tasks
- **Measurable**: Clear success/failure criteria through unit tests

Each prompt includes specific requirements, edge cases, and expected behavior to ensure thorough testing of the LLM's code generation capabilities.
