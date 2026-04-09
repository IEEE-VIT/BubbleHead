# Contributing to BubbleHead RAG

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add: your feature description"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Formatting

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```bash
pytest tests/
```

## Commit Messages

Use clear, descriptive commit messages:

- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Refactor: what was refactored`
- `Docs: documentation changes`

## Pull Request Guidelines

1. **Title**: Clear, concise description
2. **Description**: Explain what and why
3. **Testing**: Describe how you tested
4. **Screenshots**: If UI changes, include screenshots
5. **Breaking Changes**: Clearly mark any breaking changes

## Areas for Contribution

- **New document parsers** (e.g., Markdown, JSON)
- **Additional reranking algorithms**
- **Performance optimizations**
- **UI improvements**
- **Documentation enhancements**
- **Bug fixes**
- **Test coverage**

## Questions?

Open an issue with the `question` label.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

Thank you for contributing! 🎉
