# Contributing to Research Paper Tagger

Thank you for your interest in contributing to Research Paper Tagger! This document provides guidelines and information for contributors.

## How Can I Contribute?

### Reporting Bugs
- Use the GitHub issue tracker
- Include a clear description of the bug
- Provide steps to reproduce the issue
- Include error messages and logs if applicable

### Suggesting Enhancements
- Open a feature request issue
- Describe the enhancement and its benefits
- Provide use cases if possible

### Code Contributions
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Ensure all tests pass
- Submit a pull request

## Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/Sreeram5678/research-paper-tagger.git
   cd research-paper-tagger
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and concise

## Testing

- Write tests for new functionality
- Ensure all existing tests pass
- Run the test suite before submitting:
  ```bash
  python -m pytest tests/
  ```

## Pull Request Process

1. Update the README.md if needed
2. Add or update tests as appropriate
3. Ensure the code follows the project's style guidelines
4. Update documentation if adding new features
5. Submit the pull request with a clear description

## Questions?

If you have questions about contributing, feel free to:
- Open an issue on GitHub
- Contact the maintainer directly
- Check the existing documentation

Thank you for contributing to Research Paper Tagger!
